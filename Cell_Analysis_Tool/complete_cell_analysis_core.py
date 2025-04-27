# complete_cell_analysis_core.py
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent
import sys
import os
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox,
    QProgressBar, QSlider, QCheckBox, QGroupBox, QMessageBox, QTabWidget, QComboBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import tifffile as tiff
import pandas as pd
import csv

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("Warning: deep_sort_realtime not found. Install it with pip install deep-sort-realtime")
    # Provide a dummy class for development
    class DeepSort:
        def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2, nn_budget=None):
            pass
        def update_tracks(self, detections, frame=None):
            return []

from skimage import io, feature
import pyqtgraph as pg

import matplotlib.pyplot as plt
import time
# For ImageJ (if being used)
try:
    import imagej
    import scyjava
except ImportError:
    print("Warning: ImageJ libraries not found. Install with 'pip install pyimagej'")

@dataclass
class BlobParameters:
    """Class to store blob detection parameters"""
    min_sigma: float = 15
    max_sigma: float = 30
    threshold: float = 0.1
    num_sigma: int = 10

@dataclass
class TrackerParameters:
    """Class to store tracker parameters"""
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.2
    nn_budget: Optional[int] = None

@dataclass
class ProcessingParameters:
    """Parameters for blob detection and tracking"""
    blob_params: BlobParameters
    tracker_params: TrackerParameters
    top_n_cells: int = 15
    max_frames: Optional[int] = None  # Optional limit on number of frames to process

class ProgressSignal(QObject):
    """Class to emit progress updates from worker thread"""
    progress = pyqtSignal(int, str)

class BlobData:
    """Class to store information about detected blobs"""
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        frame_index: int,
        intensity_stats: dict
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.frame_index = frame_index
        self.area = np.pi * (radius ** 2)
        self.intensity_stats = intensity_stats
    
    def to_bbox(self) -> List[float]:
        """Convert blob data to bounding box format [x1, y1, width, height]"""
        return [
            self.x - self.radius,
            self.y - self.radius,
            2 * self.radius,
            2 * self.radius
        ]

class ImageProcessor:
    """Class for image processing operations"""
    def enhance_white_spots(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhance white spots in the image"""
        # Apply Gaussian blur
        kernel_size = 3
        sigma = 0.5
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        # Simple thresholding
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        
        # Dilation
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(white_mask, kernel, iterations=1)
        
        return blurred, white_mask, dilated_mask

class BlobDetector:
    """Main class for blob detection and processing"""
    def __init__(self, blob_params: BlobParameters, visualize: bool = False):
        self.blob_params = blob_params
        self.visualize = visualize
        self.processor = ImageProcessor()

    def _calculate_intensity_stats(self, frame: np.ndarray, blob: np.ndarray) -> dict:
        """Calculate intensity statistics for a blob region"""
        y, x, r = blob
        y, x = int(y), int(x)
        y_min = max(0, y - int(r))
        y_max = min(frame.shape[0], y + int(r))
        x_min = max(0, x - int(r))
        x_max = min(frame.shape[1], x + int(r))
        
        blob_region = frame[y_min:y_max, x_min:x_max]
        blob_mask = np.zeros_like(blob_region, dtype=bool)
        
        # Handle potentially empty regions
        if blob_region.shape[0] == 0 or blob_region.shape[1] == 0:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}
            
        yy, xx = np.ogrid[:blob_region.shape[0], :blob_region.shape[1]]
        mask_center_y = blob_region.shape[0] // 2
        mask_center_x = blob_region.shape[1] // 2
        
        blob_mask[(yy - mask_center_y) ** 2 + (xx - mask_center_x) ** 2 <= r ** 2] = True
        
        blob_pixels = blob_region[blob_mask]
        
        if blob_pixels.size > 0:
            return {
                'mean': np.mean(blob_pixels),
                'max': np.max(blob_pixels),
                'min': np.min(blob_pixels),
                'std': np.std(blob_pixels)
            }
        return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}

    def detect_blobs(self, mask: np.ndarray, frame_index: int, frame: np.ndarray) -> List[BlobData]:
        """Detect blobs in a single frame"""
        blobs_log = feature.blob_log(
            mask,
            min_sigma=self.blob_params.min_sigma,
            max_sigma=self.blob_params.max_sigma,
            num_sigma=self.blob_params.num_sigma,
            threshold=self.blob_params.threshold
        )
        
        blob_data_list = []
        for blob in blobs_log:
            intensity_stats = self._calculate_intensity_stats(frame, blob)
            blob_data = BlobData(
                x=blob[1],
                y=blob[0],
                radius=blob[2],
                frame_index=frame_index,
                intensity_stats=intensity_stats
            )
            blob_data_list.append(blob_data)
        
        return blob_data_list

    def create_overlay(self, frame: np.ndarray, blob_data_list: List[BlobData]) -> np.ndarray:
        """Create an overlay of detected blobs on the frame"""
        overlay = np.copy(frame)
        for blob in blob_data_list:
            cv2.circle(
                overlay,
                (int(blob.x), int(blob.y)),
                int(blob.radius * 1.3),
                (255),
                thickness=2
            )
        return overlay

class TrackingLogger:
    """Class for handling tracking data logging"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.file_handler = None
        self.writer = None
        self.headers = ['Frame', 'Cell ID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence']

    def __enter__(self):
        self.file_handler = open(self.log_file, mode='w', newline='')
        self.writer = csv.writer(self.file_handler)
        self.writer.writerow(self.headers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handler:
            self.file_handler.close()

    def log_track(self, frame_idx: int, track_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        """Log tracking data for a single track"""
        x1, y1, x2, y2 = map(int, bbox)
        self.writer.writerow([frame_idx + 1, track_id, x1, y1, x2, y2, confidence])

class LogFileProcessor:
    """Class for processing and analyzing the tracking log file."""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.df = self.load_log_file()
    
    def load_log_file(self) -> pd.DataFrame:
        """Load the log file into a Pandas DataFrame."""
        return pd.read_csv(self.log_file)
    
    def calculate_cell_frame_counts(self) -> pd.DataFrame:
        """Calculate the number of frames each cell ID appears in."""
        cell_frame_counts = self.df.groupby('Cell ID')['Frame'].nunique().reset_index()
        cell_frame_counts.columns = ['Cell ID', 'Num Frames']
        return cell_frame_counts
    
    def get_top_cells(self, top_n: int = 15) -> pd.DataFrame:
        """Get the top N cells that appear in the most frames."""
        cell_frame_counts = self.calculate_cell_frame_counts()
        top_cells = cell_frame_counts.sort_values(by='Num Frames', ascending=False).head(top_n)
        return top_cells
    
    def sort_cells_by_id(self, top_cells: pd.DataFrame) -> pd.DataFrame:
        """Sort the top cells by their original Cell ID."""
        return top_cells.sort_values(by='Cell ID')
    
    def create_id_mapping(self, sorted_top_cells: pd.DataFrame) -> dict:
        """Create a mapping from original Cell IDs to new IDs (1-N)."""
        return {old_id: new_id for new_id, old_id in enumerate(sorted_top_cells['Cell ID'], 1)}
    
    def filter_and_rename_cells(self, top_cells: pd.DataFrame, id_mapping: dict) -> pd.DataFrame:
        """Filter the original DataFrame to include only the top cells and rename the Cell IDs."""
        filtered_df = self.df[self.df['Cell ID'].isin(top_cells['Cell ID'])].copy()
        filtered_df['Cell ID'] = filtered_df['Cell ID'].map(id_mapping)
        return filtered_df

class FrameAnnotator:
    """Class to annotate frames with bounding boxes and cell IDs."""
    def __init__(self, frames: list, csv_data: pd.DataFrame):
        self.frames = frames
        self.df = csv_data
        self.grouped = self.df.groupby('Frame')
    
    def annotate_frames(self) -> list:
        """Annotate each frame with bounding boxes and cell IDs."""
        annotated_frames = []
        for frame_idx, frame in enumerate(self.frames):
            frame_number = frame_idx + 1  # Frames are 1-indexed in the CSV
            annotated_frame = frame.copy()
            
            if frame_number in self.grouped.groups:
                # Get bounding boxes and cell IDs for the current frame
                group = self.grouped.get_group(frame_number)
                for _, row in group.iterrows():
                    x1, y1 = int(row['X1']), int(row['Y1'])
                    x2, y2 = int(row['X2']), int(row['Y2'])
                    cell_id = row['Cell ID']

                    # Ensure coordinates are within frame bounds
                    height, width = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # Draw bounding box
                    # If frame is grayscale, convert to RGB for colored overlay
                    if len(annotated_frame.shape) == 2:
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2RGB)
                        
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID: {cell_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif len(annotated_frame.shape) == 2:
                # Convert grayscale frames to RGB for consistency
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2RGB)
                
            annotated_frames.append(annotated_frame)
        return annotated_frames
    


class PreprocessingPipeline:
    """Class for handling preprocessing steps (Kalman filtering and ImageJ processing)"""
    def __init__(self):
        # Initialize ImageJ

        # In PreprocessingPipeline.__init__
        try:
            self.ij = imagej.init('net.imagej:imagej')
            self.imagej_available = True
        except:
            print("Warning: ImageJ initialization failed. ImageJ processing will be disabled.")
            self.imagej_available = False
        
    def kalman_stack_filter(self, tiff_path, acquisition_noise=0.05, bias=0.8, output_path=None):
        """Apply Kalman filtering to the image stack"""
        start_time = time.time()
        print(f"Loading TIFF file: {tiff_path}")
        
        image_stack = tiff.imread(tiff_path)
        
        # Get image dimensions
        if image_stack.ndim == 3:
            n_frames, height, width = image_stack.shape
            print(f"Detected {n_frames} frames with dimensions {height}x{width}")
        else:
            raise ValueError("Input must be a 3D multi-frame TIFF file")
        
        original_dtype = image_stack.dtype
        stack_float = image_stack.astype(np.float64)
        
        if len(image_stack.shape) == 4 and image_stack.shape[3] == 3:
            print("Processing RGB stack...")
            # Split into R, G, B channels
            red = stack_float[:, :, :, 0]
            green = stack_float[:, :, :, 1]
            blue = stack_float[:, :, :, 2]
            
            # Filter each channel
            red_filtered = self._kalman_filter_grayscale(red, acquisition_noise, bias)
            green_filtered = self._kalman_filter_grayscale(green, acquisition_noise, bias)
            blue_filtered = self._kalman_filter_grayscale(blue, acquisition_noise, bias)
            
            # Merge channels back
            filtered_stack = np.stack((red_filtered, green_filtered, blue_filtered), axis=3)
        else:
            # Process grayscale stack
            filtered_stack = self._kalman_filter_grayscale(stack_float, acquisition_noise, bias)
        
        # Convert back to original data type
        filtered_stack = filtered_stack.astype(original_dtype)
        
        if output_path is None:
            input_path = Path(tiff_path)
            output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
        
        print(f"Saving filtered stack to: {output_path}")
        tiff.imwrite(str(output_path), filtered_stack)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_stack[0], cmap='gray')
        plt.title('Original (First Frame)')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_stack[1], cmap='gray')
        plt.title('Filtered (First Frame)')
        plt.colorbar()
        
        comparison_path = str(Path(output_path).parent / f"{Path(output_path).stem}_comparison.png")
        plt.savefig(comparison_path)
        print(f"Saved comparison image to: {comparison_path}")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return filtered_stack, str(output_path)
    
    def _kalman_filter_grayscale(self, stack_float, acquisition_noise, bias):
        """Apply Kalman filter to grayscale image stack"""
        n_frames, height, width = stack_float.shape
        dimension = width * height
        
        noisevar = np.full(dimension, acquisition_noise)
        predicted = stack_float[0].flatten()
        predictedvar = noisevar.copy()
        filtered_stack = stack_float.copy()
        
        for i in range(1, n_frames):
            if i % 10 == 0 or i == 1:
                print(f"Processing frame {i}/{n_frames}")
            
            observed = stack_float[i].flatten()
            kalman_gain = predictedvar / (predictedvar + noisevar)
            corrected = bias * predicted + (1.0 - bias) * observed + kalman_gain * (observed - predicted)
            correctedvar = predictedvar * (1.0 - kalman_gain)
            predictedvar = correctedvar
            predicted = corrected
            filtered_stack[i] = corrected.reshape(height, width)
        
        return filtered_stack
    
    def imagej_process(self, input_path, output_path=None):
        """Apply ImageJ processing to the filtered stack"""
        if not self.imagej_available:
            print("ImageJ is not available. Skipping ImageJ processing.")
            return input_path
            
        if output_path is None:
            output_path = str(Path(input_path).parent / f"{Path(input_path).stem}_imagej.tif")
            
        macro = """
        // Open the original image
        open("{0}");
        orig_title = getTitle();
        print("Original image: " + orig_title);

        // Duplicate the image and rename
        run("Duplicate...", "duplicate");
        dup_title = getTitle();
        print("Duplicated image: " + dup_title);
        selectWindow(dup_title);
        rename("blur_target");

        // Apply Gaussian Blur to the duplicate
        selectWindow("blur_target");
        run("Gaussian Blur...", "sigma=30 stack");

        // Subtract blurred from original
        imageCalculator("Subtract create 32-bit stack", orig_title, "blur_target");
        selectWindow("Result of " + orig_title);
        rename("subtracted");

        // Convert to 8-bit
        selectWindow("subtracted");
        setOption("ScaleConversions", true);
        run("8-bit");
        rename("bit8");

        // Duplicate the 8-bit result
        selectWindow("bit8");
        run("Duplicate...", "duplicate");
        rename("z_projection_source");

        // Z Project
        selectWindow("z_projection_source");
        run("Z Project...", "projection=[Average Intensity]");
        selectWindow("AVG_z_projection_source");
        rename("avg_projection");

        // Subtract Z-projected image from the 8-bit image
        imageCalculator("Subtract create 32-bit stack", "bit8", "avg_projection");
        selectWindow("Result of bit8");
        run("8-bit")
        rename("final_result");

        // Save the final result
        selectWindow("final_result");
        saveAs("Tiff", "{1}");

        // Arrange windows in a tiled layout for better viewing
        run("Tile");
        """.format(input_path.replace("\\", "/"), output_path.replace("\\", "/"))

        # Run the macro
        try:
            self.ij.py.run_macro(macro)
            print(f"ImageJ processing complete. Output saved at: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during ImageJ processing: {str(e)}")
            return input_path

class CellTrackingPipeline:
    """Main pipeline for cell tracking operations"""
    def __init__(self, progress_signal: ProgressSignal = None):
        self.progress_signal = progress_signal
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.preprocessing_pipeline = PreprocessingPipeline()
        
    def update_progress(self, percent: int, message: str = ""):
        """Update progress bar and message"""
        if self.progress_signal:
            self.progress_signal.progress.emit(percent, message)
            
    def process_pipeline(self, input_path: str, params: ProcessingParameters, 
                         use_kalman: bool = False, use_imagej: bool = False,
                         acquisition_noise: float = 0.05, kalman_bias: float = 0.8) -> Tuple[List[np.ndarray], str]:
        """Run the full processing pipeline"""
        self.update_progress(0, "Loading frames...")
        
        # 1. Preprocessing with Kalman filter (optional)
        processed_input_path = input_path
        
        if use_kalman:
            self.update_progress(5, "Applying Kalman filtering...")
            try:
                _, kalman_output = self.preprocessing_pipeline.kalman_stack_filter(
                    input_path,
                    acquisition_noise=acquisition_noise,
                    bias=kalman_bias, 
                    output_path=str(self.temp_dir / f"kalman_filtered.tiff")
                )
                processed_input_path = kalman_output
                self.update_progress(15, "Kalman filtering complete")
            except Exception as e:
                print(f"Error in Kalman filtering: {str(e)}")
                self.update_progress(15, "Kalman filtering failed, using original input")
                
        # 2. ImageJ processing (optional)
        if use_imagej:
            self.update_progress(20, "Applying ImageJ processing...")
            try:
                imagej_output = self.preprocessing_pipeline.imagej_process(
                    processed_input_path,
                    output_path=str(self.temp_dir / f"imagej_processed.tiff")
                )
                processed_input_path = imagej_output
                self.update_progress(30, "ImageJ processing complete")
            except Exception as e:
                print(f"Error in ImageJ processing: {str(e)}")
                self.update_progress(30, "ImageJ processing failed, using previous input")
        
        # 3. Load frames from the processed input
        try:
            tiff_frames = io.imread(processed_input_path)
            num_frames = len(tiff_frames)
            if params.max_frames:
                num_frames = min(params.max_frames, num_frames)
            tiff_frames = tiff_frames[:num_frames]
        except Exception as e:
            raise Exception(f"Error loading TIFF file: {str(e)}")
            
        self.update_progress(35, "Detecting blobs...")
        
        # 4. Detect blobs
        blob_detector = BlobDetector(params.blob_params, visualize=False)
        processor = ImageProcessor()
        
        # Create temporary blob log file
        temp_blob_log = self.temp_dir / "temp_blob_log.csv"
        
        with open(temp_blob_log, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                'Frame', 'Blob X', 'Blob Y', 'Blob Radius', 'Blob Area',
                'Blob Mean Intensity', 'Blob Max Intensity', 'Blob Min Intensity',
                'Blob Standard Deviation', 'Min Sigma', 'Max Sigma', 'Threshold'
            ])
            
            processed_frames = []
            for idx in range(num_frames):
                frame = tiff_frames[idx]
                
                # Process frame to enhance white spots
                _, _, dilated_mask = processor.enhance_white_spots(frame)
                
                # Detect blobs
                blob_data_list = blob_detector.detect_blobs(dilated_mask, idx, frame)
                
                # Create overlay
                overlay_frame = blob_detector.create_overlay(frame, blob_data_list)
                processed_frames.append(overlay_frame)
                
                # Log blob data
                for blob in blob_data_list:
                    log_writer.writerow([
                        idx + 1, blob.x, blob.y, blob.radius, blob.area,
                        blob.intensity_stats['mean'], blob.intensity_stats['max'],
                        blob.intensity_stats['min'], blob.intensity_stats['std'],
                        params.blob_params.min_sigma,
                        params.blob_params.max_sigma,
                        params.blob_params.threshold
                    ])
                    
                # Update progress
                self.update_progress(35 + (20 * idx // num_frames), f"Processing frame {idx+1}/{num_frames}...")
                
        # 5. Track cells using DeepSORT
        self.update_progress(55, "Running DeepSORT tracking...")
        
        # Create temporary tracking log
        temp_tracking_log = self.temp_dir / "temp_tracking_log.csv"
        
        # Configure DeepSORT tracker
        tracker = DeepSort(
            max_age=params.tracker_params.max_age,
            n_init=params.tracker_params.n_init,
            max_cosine_distance=params.tracker_params.max_cosine_distance,
            nn_budget=params.tracker_params.nn_budget
        )
        
        # Parse blob data for tracking
        blob_parser = pd.read_csv(temp_blob_log)
        frame_blobs = {}
        for _, row in blob_parser.iterrows():
            frame = int(row['Frame'])
            if frame not in frame_blobs:
                frame_blobs[frame] = []
                
            frame_blobs[frame].append(BlobData(
                x=float(row['Blob X']),
                y=float(row['Blob Y']),
                radius=float(row['Blob Radius']),
                frame_index=frame - 1,
                intensity_stats={
                    'mean': float(row['Blob Mean Intensity']),
                    'max': float(row['Blob Max Intensity']),
                    'min': float(row['Blob Min Intensity']),
                    'std': float(row['Blob Standard Deviation'])
                }
            ))
        
        # Run tracking
        with open(temp_tracking_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Cell ID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence'])
            
            for frame_idx, frame in enumerate(tiff_frames):
                # Skip if no blobs in this frame
                if frame_idx + 1 not in frame_blobs:
                    continue
                    
                # Prepare detections
                blobs = frame_blobs[frame_idx + 1]
                bboxes = np.array([blob.to_bbox() for blob in blobs], dtype=np.float32)
                confidences = np.ones(len(blobs), dtype=np.float32)  # Assume all detections have confidence 1.0
                
                # Convert to DeepSORT format [x1, y1, w, h]
                ds_detections = [(
                    [b[0], b[1], b[2], b[3]],  # x, y, width, height
                    conf
                ) for b, conf in zip(bboxes, confidences)]
                
                if len(ds_detections) > 0:
                    # Convert grayscale to RGB if needed
                    if len(frame.shape) == 2 or (len(frame.shape) > 2 and frame.shape[2] == 1):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:
                        frame_rgb = frame
                    
                    tracks = tracker.update_tracks(ds_detections, frame=frame_rgb)
                    
                    # Log tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                            
                        ltrb = track.to_ltrb()  # left, top, right, bottom
                        writer.writerow([
                            frame_idx + 1,  # Frame (1-indexed)
                            track.track_id,  # Cell ID
                            ltrb[0],  # X1
                            ltrb[1],  # Y1
                            ltrb[2],  # X2
                            ltrb[3],  # Y2
                            1.0  # Confidence
                        ])
                        
                # Update progress
                self.update_progress(55 + (25 * frame_idx // num_frames), f"Tracking frame {frame_idx+1}/{num_frames}...")
                        
        # 6. Process tracking log to get top cells
        self.update_progress(80, "Processing tracking results...")
        
        # Get top N cells
        log_processor = LogFileProcessor(temp_tracking_log)
        top_cells = log_processor.get_top_cells(params.top_n_cells)
        sorted_top_cells = log_processor.sort_cells_by_id(top_cells)
        id_mapping = log_processor.create_id_mapping(sorted_top_cells)
        filtered_df = log_processor.filter_and_rename_cells(top_cells, id_mapping)
        
        # Save top cells data
        output_csv = str(self.temp_dir / f"top_{params.top_n_cells}_cells.csv")
        filtered_df.to_csv(output_csv, index=False)
        
        # 7. Annotate frames with cell IDs
        self.update_progress(90, "Creating final visualization...")
        
        annotator = FrameAnnotator(tiff_frames, filtered_df)
        annotated_frames = annotator.annotate_frames()
        
        # 8. Save annotated frames
        output_tiff = str(self.temp_dir / "annotated_output.tiff")
        tiff.imwrite(output_tiff, np.array(annotated_frames))
        
        self.update_progress(100, "Processing complete!")
        
        return annotated_frames, output_csv

class ImageViewer(QWidget):
    """Custom widget for displaying TIFF frames with overlay"""
    frame_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create top section with image and graphs side by side
        top_section = QHBoxLayout()
        
        # Left side - Image display
        image_section = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        image_section.addWidget(self.image_label)
        
        # Add image section to top section
        top_section.addLayout(image_section, 2)  # Give more space to image
        
        # Right side - Graphs
        graphs_section = QVBoxLayout()
        self.setup_graphs()
        graphs_section.addLayout(self.plots_layout)
        
        # Add graphs section to top section
        top_section.addLayout(graphs_section, 1)  # Give less space to graphs
        
        # Add top section to main layout
        self.layout.addLayout(top_section)
        
        # Bottom section - Controls
        controls_section = QVBoxLayout()
        
        # Frame info
        self.frame_info = QLabel("Frame: 0 / 0")
        controls_section.addWidget(self.frame_info)
        
        # Playback controls
        self.controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.set_frame)
        
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.frame_slider)
        controls_section.addLayout(self.controls_layout)
        
        # Add controls section to main layout
        self.layout.addLayout(controls_section)
        
        # Timer for playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.frames = []
        self.current_frame = 0
        self.playing = False
        
        # Add data storage
        self.current_cell_data = {
            'time': [],
            'x_pos': [],
            'y_pos': [],
            'velocity': [],
            'acceleration': [],
            'area': []
        }
    
    def setup_graphs(self):
        """Setup the graphs layout"""
        self.plots_layout = QVBoxLayout()
        
        # Create plot widgets with dark background
        self.position_plot = self.create_plot_widget("Position")
        self.path_plot = self.create_plot_widget("Path")
        self.velocity_plot = self.create_plot_widget("Velocity")
        self.acceleration_plot = self.create_plot_widget("Acceleration")
        
        # Configure plots for dark theme
        for plot in [self.position_plot, self.path_plot, self.velocity_plot, self.acceleration_plot]:
            plot.setBackground('k')  # Black background
            plot.getAxis('bottom').setPen('w')  # White axes
            plot.getAxis('left').setPen('w')
            plot.getAxis('bottom').setTextPen('w')  # White text
            plot.getAxis('left').setTextPen('w')
            plot.setTitle(title=plot.plotItem.titleLabel.text, color='w')  # White title
            plot.showGrid(x=True, y=True, alpha=0.3)  # Subtle grid
            
            # Set fixed size for plots
            plot.setFixedHeight(120)  # Adjust this value as needed
        
        # Add plots to layout
        self.plots_layout.addWidget(self.position_plot)
        self.plots_layout.addWidget(self.path_plot)
        self.plots_layout.addWidget(self.velocity_plot)
        self.plots_layout.addWidget(self.acceleration_plot)
    
    def create_plot_widget(self, title):
        """Create a pyqtgraph plot widget"""
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setLabel('bottom', 'Frame')
        plot_widget.setLabel('left', 'Value')
        plot_widget.addLegend()
        plot_widget.setMouseEnabled(x=True, y=True)
        return plot_widget
    
    def load_frames(self, frames: List[np.ndarray]):
        """Load frames for display"""
        self.frames = frames
        self.frame_slider.setMaximum(len(frames) - 1)
        self.frame_info.setText(f"Frame: 1 / {len(frames)}")
        self.show_frame(0)
    
    def show_frame(self, index: int):
        """Display a specific frame"""
        if not self.frames or index >= len(self.frames):
            return
            
        frame = self.frames[index]
        
        # Convert to 8-bit if needed
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
        # Update frame info
        self.frame_info.setText(f"Frame: {index + 1} / {len(self.frames)}")
        
        # Update plots for current frame
        self.update_plots_for_frame(index + 1)
        
        # Emit frame change signal
        self.frame_changed.emit(index + 1)
    
    def update_plots_for_frame(self, frame_number):
        """Update plots up to the current frame"""
        if not self.current_cell_data['time']:
            return
            
        # Find the index for the current frame
        try:
            current_idx = self.current_cell_data['time'].index(frame_number)
        except ValueError:
            return
            
        # Get data up to current frame
        times = self.current_cell_data['time'][:current_idx + 1]
        x_pos = self.current_cell_data['x_pos'][:current_idx + 1]
        y_pos = self.current_cell_data['y_pos'][:current_idx + 1]
        velocities = self.current_cell_data['velocity'][:current_idx + 1]
        accelerations = self.current_cell_data['acceleration'][:current_idx + 1]
        
        # Update plots
        self.position_plot.clear()
        self.position_plot.plot(times, x_pos, pen='b', name='X Position')
        self.position_plot.plot(times, y_pos, pen='r', name='Y Position')
        
        self.path_plot.clear()
        self.path_plot.plot(x_pos, y_pos, pen='g', name='Path')
        
        self.velocity_plot.clear()
        if velocities:
            self.velocity_plot.plot(times, velocities, pen='y', name='Velocity')
        
        self.acceleration_plot.clear()
        if accelerations:
            self.acceleration_plot.plot(times, accelerations, pen='m', name='Acceleration')
        
        # Auto-range plots to show full data
        for plot in [self.position_plot, self.path_plot, self.velocity_plot, self.acceleration_plot]:
            plot.enableAutoRange()
    
    def next_frame(self):
        """Show next frame during playback"""
        if not self.frames:
            return
            
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.frame_slider.setValue(self.current_frame)
        self.show_frame(self.current_frame)
    
    def toggle_playback(self):
        """Toggle play/pause"""
        self.playing = not self.playing
        if self.playing:
            self.play_timer.start(50)  # 20 fps
            self.play_button.setText("Pause")
        else:
            self.play_timer.stop()
            self.play_button.setText("Play")
    
    def set_frame(self, value: int):
        """Set current frame from slider"""
        self.current_frame = value
        self.show_frame(value)

    def update_cell_data(self, cell_data):
        """Update the stored cell data and refresh plots"""
        self.current_cell_data = cell_data
        self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data"""
        if not self.current_cell_data['time']:
            return
            
        times = self.current_cell_data['time']
        x_pos = self.current_cell_data['x_pos']
        y_pos = self.current_cell_data['y_pos']
        velocities = self.current_cell_data['velocity']
        accelerations = self.current_cell_data['acceleration']
        
        self.position_plot.clear()
        self.position_plot.plot(times, x_pos, pen='b', name='X Position')
        self.position_plot.plot(times, y_pos, pen='r', name='Y Position')
        
        self.path_plot.clear()
        self.path_plot.plot(x_pos, y_pos, pen='g', name='Path')
        
        self.velocity_plot.clear()
        if velocities:
            self.velocity_plot.plot(times, velocities, pen='y', name='Velocity')
        
        self.acceleration_plot.clear()
        if accelerations:
            self.acceleration_plot.plot(times, accelerations, pen='m', name='Acceleration')
        
        # Auto-range plots to show full data
        for plot in [self.position_plot, self.path_plot, self.velocity_plot, self.acceleration_plot]:
            plot.enableAutoRange()

class CellMetricsViewer(QWidget):
    """Widget for displaying real-time cell metrics and statistics"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Cell selection and playback controls
        controls_layout = QHBoxLayout()
        
        # Cell selection
        self.cell_combo = QComboBox()
        self.cell_combo.currentIndexChanged.connect(self.on_cell_selected)
        controls_layout.addWidget(QLabel("Select Cell:"))
        controls_layout.addWidget(self.cell_combo)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.frame_slider)
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0 / 0")
        controls_layout.addWidget(self.frame_label)
        
        self.layout.addLayout(controls_layout)
        
        # Statistics panel
        stats_group = QGroupBox("Real-time Cell Statistics")
        stats_layout = QVBoxLayout()
        self.stats_labels = {
            'current_frame': QLabel("Current Frame: -"),
            'position': QLabel("Position (x, y): -, -"),
            'velocity': QLabel("Current Velocity: -"),
            'acceleration': QLabel("Current Acceleration: -"),
            'area': QLabel("Current Area: -"),
            'total_distance': QLabel("Total Distance: -"),
            'direction': QLabel("Movement Direction: -")
        }
        for label in self.stats_labels.values():
            stats_layout.addWidget(label)
        stats_group.setLayout(stats_layout)
        self.layout.addWidget(stats_group)
        
        # Create real-time plots
        self.setup_real_time_plots()
        
        # Initialize data storage
        self.df = None
        self.current_cell_id = None
        self.current_frame = 0
        self.max_frame = 0
        
        # Full data buffers for complete history
        self.full_data = {
            'time': [],
            'x_pos': [],
            'y_pos': [],
            'velocity': [],
            'acceleration': [],
            'area': []
        }
        
    def setup_real_time_plots(self):
        """Setup real-time plotting widgets"""
        plots_layout = QHBoxLayout()
        
        # Create two columns of plots
        left_plots = QVBoxLayout()
        right_plots = QVBoxLayout()
        
        # Initialize plot widgets with dark background
        self.position_plot = self.create_plot_widget("Position Tracking")
        self.path_plot = self.create_plot_widget("Movement Path")
        self.velocity_plot = self.create_plot_widget("Velocity")
        self.acceleration_plot = self.create_plot_widget("Acceleration")
        
        # Configure plots for dark theme
        for plot in [self.position_plot, self.path_plot, self.velocity_plot, self.acceleration_plot]:
            plot.setBackground('k')  # Black background
            plot.getAxis('bottom').setPen('w')  # White axes
            plot.getAxis('left').setPen('w')
            plot.getAxis('bottom').setTextPen('w')  # White text
            plot.getAxis('left').setTextPen('w')
            plot.setTitle(title=plot.plotItem.titleLabel.text, color='w')  # White title
            plot.showGrid(x=True, y=True, alpha=0.3)  # Subtle grid
        
        # Add plots to columns
        left_plots.addWidget(self.position_plot)
        left_plots.addWidget(self.velocity_plot)
        right_plots.addWidget(self.path_plot)
        right_plots.addWidget(self.acceleration_plot)
        
        plots_layout.addLayout(left_plots)
        plots_layout.addLayout(right_plots)
        self.layout.addLayout(plots_layout)
        
    def create_plot_widget(self, title):
        """Create a pyqtgraph plot widget"""
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setLabel('bottom', 'Frame')
        plot_widget.setLabel('left', 'Value')
        plot_widget.addLegend()
        # Enable mouse interaction for zooming and panning
        plot_widget.setMouseEnabled(x=True, y=True)
        return plot_widget
    
    def load_data(self, df: pd.DataFrame):
        """Load tracking data and update cell selection"""
        self.df = df
        self.cell_combo.clear()
        
        # Get unique cell IDs and sort them
        cell_ids = sorted(df['Cell ID'].unique())
        for cell_id in cell_ids:
            self.cell_combo.addItem(f"Cell {cell_id}")
        
        # Set up slider range
        self.max_frame = df['Frame'].max()
        self.frame_slider.setRange(1, self.max_frame)
        self.frame_slider.setValue(1)
        self.frame_label.setText(f"Frame: 1 / {self.max_frame}")
        
        # Clear data
        self.clear_data()
    
    def clear_data(self):
        """Clear all data buffers and plots"""
        self.full_data = {key: [] for key in self.full_data}
        self.clear_plots()
    
    def clear_plots(self):
        """Clear all plots"""
        self.position_plot.clear()
        self.path_plot.clear()
        self.velocity_plot.clear()
        self.acceleration_plot.clear()
    
    def on_cell_selected(self):
        """Handle cell selection change"""
        if self.df is None:
            return
        
        self.current_cell_id = int(self.cell_combo.currentText().split()[1])
        self.clear_data()
        
        # Pre-calculate all metrics for the selected cell
        self.calculate_full_metrics()
        
        # Update the ImageViewer with the full data
        if hasattr(self.parent(), 'viewer'):
            self.parent().viewer.update_cell_data(self.full_data)
        
        # Update display for current frame
        self.update_frame(self.frame_slider.value())
    
    def calculate_full_metrics(self):
        """Calculate metrics for all frames of the selected cell"""
        if self.df is None or self.current_cell_id is None:
            return
        
        # Get all data for the selected cell
        cell_data = self.df[self.df['Cell ID'] == self.current_cell_id].sort_values('Frame')
        
        for _, row in cell_data.iterrows():
            frame = row['Frame']
            x = (row['X1'] + row['X2']) / 2
            y = (row['Y1'] + row['Y2']) / 2
            area = (row['X2'] - row['X1']) * (row['Y2'] - row['Y1'])
            
            self.full_data['time'].append(frame)
            self.full_data['x_pos'].append(x)
            self.full_data['y_pos'].append(y)
            self.full_data['area'].append(area)
        
        # Calculate velocities and accelerations
        if len(self.full_data['x_pos']) >= 2:
            dx = np.diff(self.full_data['x_pos'])
            dy = np.diff(self.full_data['y_pos'])
            velocities = np.sqrt(dx**2 + dy**2)
            self.full_data['velocity'] = [0] + list(velocities)  # Add 0 for first frame
            
            if len(velocities) >= 2:
                accelerations = np.diff(velocities)
                self.full_data['acceleration'] = [0, 0] + list(accelerations)  # Add 0s for first two frames
    
    def on_slider_changed(self, frame):
        """Handle slider value change"""
        self.update_frame(frame)
        self.frame_label.setText(f"Frame: {frame} / {self.max_frame}")
    
    def update_frame(self, frame_number):
        """Update display for current frame"""
        if not self.full_data['time']:
            return
        
        # Find the index for the current frame
        try:
            current_idx = self.full_data['time'].index(frame_number)
        except ValueError:
            return
        
        # Update plots with full history up to current frame
        self.update_plots(current_idx)
        
        # Update statistics
        self.update_statistics(
            frame_number,
            self.full_data['x_pos'][current_idx],
            self.full_data['y_pos'][current_idx],
            self.full_data['velocity'][current_idx] if current_idx < len(self.full_data['velocity']) else 0,
            self.full_data['acceleration'][current_idx] if current_idx < len(self.full_data['acceleration']) else 0,
            self.full_data['area'][current_idx]
        )
    
    def update_plots(self, current_idx):
        """Update all plots with data up to current index"""
        # Get data up to current frame
        times = self.full_data['time'][:current_idx + 1]
        x_pos = self.full_data['x_pos'][:current_idx + 1]
        y_pos = self.full_data['y_pos'][:current_idx + 1]
        velocities = self.full_data['velocity'][:current_idx + 1]
        accelerations = self.full_data['acceleration'][:current_idx + 1]
        
        # Update plots in this viewer
        self.position_plot.clear()
        self.position_plot.plot(times, x_pos, pen='b', name='X Position')
        self.position_plot.plot(times, y_pos, pen='r', name='Y Position')
        
        self.path_plot.clear()
        self.path_plot.plot(x_pos, y_pos, pen='g', name='Path')
        
        self.velocity_plot.clear()
        if velocities:
            self.velocity_plot.plot(times, velocities, pen='y', name='Velocity')
        
        self.acceleration_plot.clear()
        if accelerations:
            self.acceleration_plot.plot(times, accelerations, pen='m', name='Acceleration')
        
        # Auto-range plots to show full data
        for plot in [self.position_plot, self.path_plot, self.velocity_plot, self.acceleration_plot]:
            plot.enableAutoRange()
        
        # Update the ImageViewer plots if it exists
        if hasattr(self.parent(), 'viewer'):
            current_data = {
                'time': times,
                'x_pos': x_pos,
                'y_pos': y_pos,
                'velocity': velocities,
                'acceleration': accelerations,
                'area': self.full_data['area'][:current_idx + 1]
            }
            self.parent().viewer.update_cell_data(current_data)
    
    def update_statistics(self, frame, x, y, velocity, acceleration, area):
        """Update real-time statistics display"""
        self.stats_labels['current_frame'].setText(f"Current Frame: {frame}")
        self.stats_labels['position'].setText(f"Position (x, y): {x:.1f}, {y:.1f}")
        self.stats_labels['velocity'].setText(f"Current Velocity: {velocity:.2f} pixels/frame")
        self.stats_labels['acceleration'].setText(f"Current Acceleration: {acceleration:.2f} pixels/frame²")
        self.stats_labels['area'].setText(f"Current Area: {area:.1f} pixels²")
        
        if len(self.full_data['x_pos']) > 1:
            total_distance = sum(
                np.sqrt(
                    (x2-x1)**2 + (y2-y1)**2
                ) for (x1, y1), (x2, y2) in zip(
                    zip(self.full_data['x_pos'][:-1], self.full_data['y_pos'][:-1]),
                    zip(self.full_data['x_pos'][1:], self.full_data['y_pos'][1:])
                )
            )
            self.stats_labels['total_distance'].setText(f"Total Distance: {total_distance:.1f} pixels")
            
            # Calculate movement direction
            if len(self.full_data['x_pos']) >= 2:
                dx = self.full_data['x_pos'][-1] - self.full_data['x_pos'][-2]
                dy = self.full_data['y_pos'][-1] - self.full_data['y_pos'][-2]
                angle = np.degrees(np.arctan2(dy, dx))
                self.stats_labels['direction'].setText(f"Movement Direction: {angle:.1f}°")