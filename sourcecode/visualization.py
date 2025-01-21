import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Visualizer:
    @staticmethod
    def visualize_masks(image, masks, output_path):
        """Visualize the masks on the original image."""
        if not masks:
            return
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        
        colors = ['#4169E1', '#FF6B6B', '#FFD700']  # blue, red, yellow
        
        for idx, mask in enumerate(tqdm(masks, desc="Visualizing masks", leave=False)):
            color = colors[idx % len(colors)]
            mask_image = mask['segmentation']
            masked_area = np.ma.masked_where(~mask_image, mask_image)
            plt.imshow(masked_area, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() 