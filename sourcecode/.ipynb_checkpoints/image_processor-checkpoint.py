import cv2
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class ImageProcessor:
    def __init__(self, logger):
        self.logger = logger

    def display_image(self, title, image):
        plt.figure(figsize=(10, 8))
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:  # RGB
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('on')
        plt.show()

    def process_frame(self, img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        _, binary_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        processed_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return processed_frame, contours

    def draw_tracking(self, frame, tracks, target_cell_id=1):
        for blob_id, positions in tracks.items():
            if blob_id == target_cell_id:
                for i in range(1, len(positions)):
                    x1, y1, _ = positions[i - 1]
                    x2, y2, _ = positions[i]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 0, 0), 2)

                x, y, _ = positions[-1]
                cv2.putText(frame, f"ID {blob_id}", (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.rectangle(frame, (int(x) - 15, int(y) - 15),
                              (int(x) + 15, int(y) + 15), (0, 255, 0), 2)
        return frame

    def process_tiff_stack(self, input_path, output_path, tracker, metrics_calculator):
        try:
            processed_frames = []
            with tifffile.TiffFile(input_path) as tif:
                num_frames = len(tif.pages)
                self.logger.info(f"Total frames in TIFF: {num_frames}")

                # Display input first frame
                first_frame = tif.pages[0].asarray()
                self.display_image("Input First Frame", first_frame)

                for frame_index, frame in enumerate(tqdm(tif.pages, desc="Processing frames")):
                    img = frame.asarray()
                    processed_frame, contours = self.process_frame(img)
                    tracker.update_tracks(contours, frame_index)
                    processed_frame = self.draw_tracking(processed_frame, tracker.get_tracks())
                    processed_frames.append(processed_frame)

                    if frame_index == 0:
                        self.display_image("Processed First Frame with Tracking", processed_frame)

            # Save processed frames
            self.logger.info("Saving processed frames...")
            processed_stack = np.stack(processed_frames)
            tifffile.imwrite(output_path, processed_stack, photometric='rgb')

            return num_frames

        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise
