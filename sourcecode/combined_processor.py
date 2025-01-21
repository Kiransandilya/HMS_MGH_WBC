import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm

class CombinedProcessor:
    def __init__(self):
        self.min_sigma = 10
        self.max_sigma = 30
        self.threshold = 0.1
        
    def enhance_white_spots(self, img):
        """Enhance white spots in the image and return both enhanced and RGB versions."""
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # Create mask for white spots using threshold
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        # Dilate the white spots slightly
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        
        # Convert to RGB for SAM
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return white_mask, rgb_frame
    
    def detect_blobs(self, frame):
        """Detect blobs using OpenCV."""
        # Enhance white spots
        white_mask, _ = self.enhance_white_spots(frame)
        
        # Detect blobs
        blobs = feature.blob_log(
            white_mask,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=10,
            threshold=self.threshold
        )
        
        # Create blob mask
        blob_mask = np.zeros_like(frame, dtype=np.uint8)
        for blob in blobs:
            y, x, r = blob
            # Back to original size
            cv2.circle(blob_mask, (int(x), int(y)), int(r * 2), 255, -1)
        
        return blob_mask, blobs
    
    def validate_sam_masks(self, sam_masks, blob_mask):
        """Validate SAM masks against blob detection."""
        validated_masks = []
        
        if not sam_masks:  # Handle case where SAM returns no masks
            return validated_masks
            
        for mask in sam_masks:
            # Get the segmentation mask
            seg_mask = mask['segmentation'].astype(np.uint8) * 255
            
            # Calculate intersection with blob mask
            intersection = cv2.bitwise_and(seg_mask, blob_mask)
            
            # Calculate IoU
            intersection_area = np.sum(intersection > 0)
            sam_area = np.sum(seg_mask > 0)
            
            if sam_area > 0:  # Prevent division by zero
                # If significant overlap, keep the mask
                if intersection_area / sam_area > 0.3:  # Adjust threshold as needed
                    validated_masks.append(mask)
        
        return validated_masks
    
    def create_combined_visualization(self, frame, sam_masks, blobs):
        """Create visualization with both SAM masks and blob detections."""
        # Create RGB visualization
        vis_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Draw blob circles
        for blob in blobs:
            y, x, r = blob
            cv2.circle(vis_img, (int(x), int(y)), int(r * 2), (0, 255, 0), 2)
        
        # Draw SAM masks
        for idx, mask in enumerate(sam_masks):
            color = [(255, 0, 0), (0, 0, 255), (255, 255, 0)][idx % 3]
            mask_overlay = np.zeros_like(vis_img)
            mask_overlay[mask['segmentation']] = color
            vis_img = cv2.addWeighted(vis_img, 1, mask_overlay, 0.4, 0)
        
        return vis_img
    
    def get_blob_info(self, frame):
        """Get blob detection info including coordinates and area."""
        # Enhance white spots
        white_mask, _ = self.enhance_white_spots(frame)
        
        # Detect blobs
        blobs = feature.blob_log(
            white_mask,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=10,
            threshold=self.threshold
        )
        
        # Calculate blob information
        blob_info = []
        for blob in blobs:
            y, x, r = blob
            area = np.pi * (r * 2) ** 2  # Area of circle
            blob_info.append({
                'x': int(x),
                'y': int(y),
                'radius': r,
                'area': area
            })
        
        return blob_info
    
    def get_sam_mask_info(self, masks):
        """Get SAM mask information including centroid and area."""
        mask_info = []
        if not masks:
            return mask_info
            
        for mask in masks:
            segmentation = mask['segmentation']
            # Calculate centroid
            y_coords, x_coords = np.where(segmentation)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                area = np.sum(segmentation)
                
                mask_info.append({
                    'x': centroid_x,
                    'y': centroid_y,
                    'area': area,
                    'mask': mask
                })
        
        return mask_info
    
    def is_mask_inside_blob(self, mask_info, blob_info, overlap_threshold=0.5):
        """Check if a SAM mask is inside and nearby a blob."""
        mask_center = np.array([mask_info['x'], mask_info['y']])
        blob_center = np.array([blob_info['x'], blob_info['y']])
        
        # Calculate distance between centers
        distance = np.linalg.norm(mask_center - blob_center)
        
        # Calculate size-based criteria
        blob_radius = blob_info['radius'] * 2
        mask_radius = np.sqrt(mask_info['area'] / np.pi)  # Approximate mask radius
        
        # Rules for mask to be considered valid:
        # 1. Center must be within blob radius
        center_inside = distance <= blob_radius
        
        # 2. Mask size should be comparable to blob size
        size_ratio = mask_radius / blob_radius
        size_appropriate = 0.5 <= size_ratio <= 1.5  # Mask should be between 0.5x and 2x blob size
        
        # 3. Distance between centers should be small relative to blob size
        distance_ratio = distance / blob_radius
        centers_close = distance_ratio <= 0.7  # Centers should be within 70% of blob radius
        
        # All conditions must be met
        return center_inside and size_appropriate and centers_close
    
    def process_and_log(self, frame, sam_masks):
        """Process frame with both methods and return detailed logs."""
        # Convert frame to RGB for SAM if it's grayscale
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame
        
        # Get blob information
        blob_info = self.get_blob_info(frame)
        
        # Get SAM mask information
        mask_info = self.get_sam_mask_info(sam_masks)
        
        # Find overlapping masks
        valid_masks = []
        overlap_log = []
        
        for mask in mask_info:
            for blob in blob_info:
                if self.is_mask_inside_blob(mask, blob):
                    valid_masks.append(mask['mask'])
                    overlap_log.append({
                        'mask_x': mask['x'],
                        'mask_y': mask['y'],
                        'mask_area': mask['area'],
                        'blob_x': blob['x'],
                        'blob_y': blob['y'],
                        'blob_area': blob['area'],
                        'blob_radius': blob['radius']
                    })
                    break  # One match is enough
        
        return {
            'blob_info': blob_info,
            'mask_info': mask_info,
            'overlap_info': overlap_log,
            'valid_masks': valid_masks
        }
    
    def create_visualizations(self, frame, process_info):
        """Create separate visualizations for blobs, SAM, and combined."""
        # Convert frame to RGB if it's grayscale
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame
        
        # Original with blobs
        blob_vis = frame_rgb.copy()
        for blob in process_info['blob_info']:
            cv2.circle(blob_vis, (blob['x'], blob['y']), 
                      int(blob['radius'] * 2), (0, 255, 0), 2)
        
        # Original with all SAM masks
        sam_vis = frame_rgb.copy()
        for mask_info in process_info['mask_info']:
            mask = mask_info['mask']
            color = (255, 0, 0)  # Blue for all masks
            mask_overlay = np.zeros_like(sam_vis)
            mask_overlay[mask['segmentation']] = color
            sam_vis = cv2.addWeighted(sam_vis, 1, mask_overlay, 0.4, 0)
        
        # Original with validated masks
        combined_vis = frame_rgb.copy()
        for mask in process_info['valid_masks']:
            color = (0, 0, 255)  # Red for valid masks
            mask_overlay = np.zeros_like(combined_vis)
            mask_overlay[mask['segmentation']] = color
            combined_vis = cv2.addWeighted(combined_vis, 1, mask_overlay, 0.4, 0)
        
        return blob_vis, sam_vis, combined_vis 