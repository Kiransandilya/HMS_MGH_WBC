import cv2
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm

class ImageProcessor:
    @staticmethod
    def read_tiff(file_path):
        """Read a TIFF file and return its frames."""
        print(f"Reading TIFF file: {file_path}")
        tiff_data = tifffile.imread(file_path)
        print(f"Found {len(tiff_data)} frames")
        return tiff_data
    
    @staticmethod
    def preprocess_frame(frame):
        """Preprocess a single frame."""
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame)
        
        # Convert to RGB (required for SAM)
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return frame, image_rgb

    @staticmethod
    def filter_masks(masks, original_image):
        """Filter masks based on brightness and area."""
        filtered_masks = []
        mask_list = []
        
        for mask in tqdm(masks, desc="Filtering masks", leave=False):
            segmentation = mask['segmentation']
            masked_image = original_image * segmentation
            mean_brightness = np.sum(masked_image) / np.sum(segmentation) if np.sum(segmentation) > 0 else 0
            
            area = np.sum(segmentation)
            if 50 < area < 3000 and mean_brightness > 100:
                mask_list.append((mask, mean_brightness))
        
        mask_list.sort(key=lambda x: x[1], reverse=True)
        filtered_masks = [item[0] for item in mask_list[:3]]
        return filtered_masks 
  