import cv2
import numpy as np
import torch
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Dict, Any, Optional
import tifffile

class SAMProcessor:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"):
        """Initialize the SAM processor."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sam = self._initialize_model(checkpoint_path, model_type)
        self.mask_generator = self._configure_mask_generator()
        
    def _initialize_model(self, checkpoint_path: str, model_type: str):
        """Initialize the SAM model."""
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        return sam
        
    def _configure_mask_generator(self):
        """Configure the automatic mask generator."""
        return SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=50,
            output_mode="binary_mask"
        )
        
    def load_tiff_frame(self, tiff_path: str, frame_idx: int) -> np.ndarray:
        """Load a specific frame from a TIFF file."""
        with tifffile.TiffFile(tiff_path) as tif:
            if frame_idx >= len(tif.pages):
                raise ValueError(f"Frame index {frame_idx} exceeds total frames {len(tif.pages)}")
            return tif.pages[frame_idx].asarray()
            
    def preprocess_image(self, image: np.ndarray) -> tuple:
        """Preprocess the input image."""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Convert to RGB (required for SAM)
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return image, image_rgb
        
    def filter_masks(self, masks: List[Dict], original_image: np.ndarray) -> List[Dict]:
        """Filter masks based on brightness and area."""
        mask_list = []
        for mask in masks:
            segmentation = mask['segmentation']
            masked_image = original_image * segmentation
            mean_brightness = np.sum(masked_image) / np.sum(segmentation) if np.sum(segmentation) > 0 else 0
            
            area = np.sum(segmentation)
            if 50 < area < 3000 and mean_brightness > 100:
                mask_list.append((mask, mean_brightness))
                
        # Sort by brightness and take top 3
        mask_list.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in mask_list[:3]]
        
    def process_frame(self, image: np.ndarray) -> tuple:
        """Process a single frame and return the original image and filtered masks."""
        original, image_rgb = self.preprocess_image(image)
        masks = self.mask_generator.generate(image_rgb)
        filtered_masks = self.filter_masks(masks, original)
        return original, filtered_masks
