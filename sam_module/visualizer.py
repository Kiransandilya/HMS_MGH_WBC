import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import logging

class SAMVisualizer:
    def __init__(self):
        """Initialize the visualizer with default colors."""
        self.colors = ['#4169E1', '#FF6B6B', '#FFD700']  # blue, red, yellow
        
    def visualize_masks(self, image: np.ndarray, masks: List[Dict], 
                       figsize: tuple = (10, 10)) -> plt.Figure:
        """Visualize masks on the original image."""
        if not masks:
            logging.info("No masks to visualize")
            return None
            
        fig = plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        
        for idx, mask in enumerate(masks):
            color = self.colors[idx % len(self.colors)]
            mask_image = mask['segmentation']
            masked_area = np.ma.masked_where(~mask_image, mask_image)
            plt.imshow(masked_area, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))
            
        plt.axis('off')
        plt.tight_layout()
        return fig
        
    def save_visualization(self, fig: plt.Figure, output_path: str):
        """Save the visualization to a file."""
        if fig is not None:
            fig.savefig(output_path)
            plt.close(fig)
            
    def display_multiple_frames(self, images: List[np.ndarray], masks_list: List[List[Dict]], 
                              num_cols: int = 3, figsize: tuple = (15, 15)):
        """Display multiple frames with their masks in a grid."""
        num_frames = len(images)
        num_rows = (num_frames + num_cols - 1) // num_cols
        
        fig = plt.figure(figsize=figsize)
        for idx in range(num_frames):
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.imshow(images[idx], cmap='gray')
            
            for mask_idx, mask in enumerate(masks_list[idx]):
                color = self.colors[mask_idx % len(self.colors)]
                mask_image = mask['segmentation']
                masked_area = np.ma.masked_where(~mask_image, mask_image)
                plt.imshow(masked_area, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))
                
            plt.axis('off')
            
        plt.tight_layout()
        return fig
