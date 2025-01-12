import os
import cv2
import numpy as np
import torch
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def initialize_sam(checkpoint_path, model_type="vit_h", device="cuda"):
    """Initialize the SAM model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return sam


def preprocess_image(image_path):
    """Preprocess the input image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Convert to RGB (required for SAM)
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return image, image_rgb


def filter_masks(masks, original_image):
    """Filter masks based on brightness and area."""
    filtered_masks = []

    # Convert masks to list for sorting
    mask_list = []
    for mask in masks:
        segmentation = mask['segmentation']
        # Calculate mean brightness in the mask region
        masked_image = original_image * segmentation
        mean_brightness = np.sum(masked_image) / np.sum(segmentation) if np.sum(segmentation) > 0 else 0

        area = np.sum(segmentation)
        if 50 < area < 3000 and mean_brightness > 100:  # Adjusted thresholds
            mask_list.append((mask, mean_brightness))

    # Sort by brightness
    mask_list.sort(key=lambda x: x[1], reverse=True)

    # Take top 3 brightest masks
    filtered_masks = [item[0] for item in mask_list[:3]]
    return filtered_masks


def visualize_results(image, masks, output_path):
    """Visualize the masks on the original image."""
    if not masks:
        logging.info("No masks to visualize")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')

    colors = ['#4169E1', '#FF6B6B', '#FFD700']  # blue, red, yellow

    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        mask_image = mask['segmentation']
        masked_area = np.ma.masked_where(~mask_image, mask_image)
        plt.imshow(masked_area, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))

    plt.axis('off')
    plt.tight_layout()

    # Save the output
    plt.savefig(output_path)
    plt.close()


def process_image(image_path, output_folder, sam, mask_generator):
    """Process a single image and save results."""
    start_time = time.time()

    original, image_rgb = preprocess_image(image_path)

    masks = mask_generator.generate(image_rgb)
    logging.info(f"Found {len(masks)} initial masks for {image_path}")

    filtered_masks = filter_masks(masks, original)
    logging.info(f"Selected {len(filtered_masks)} final masks for {image_path}")

    output_path = os.path.join(output_folder, os.path.basename(image_path).replace(".png", "_output.png"))
    visualize_results(original, filtered_masks, output_path)

    elapsed_time = time.time() - start_time
    logging.info(f"Processed {image_path} in {elapsed_time:.2f} seconds")


def main():
    # Set up paths
    input_folder = "wbc_frames_100"
    output_folder = "processed"
    sam_checkpoint = "sam_vit_h_4b8939.pth"

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize SAM
    logging.info("Initializing SAM model...")
    sam = initialize_sam(sam_checkpoint)

    # Configure automatic mask generator
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=50,
        output_mode="binary_mask"
    )

    # Process all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    logging.info(f"Found {len(image_files)} images in the input folder")

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        process_image(image_path, output_folder, sam, mask_generator)

    logging.info("Processing completed for all images")


if __name__ == "__main__":
    main()
