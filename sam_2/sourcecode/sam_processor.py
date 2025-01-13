import torch
import urllib.request
import ssl
import certifi
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from .config import SAM_MODELS, SAM_WEIGHTS_DIR

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class SAMProcessor:
    def __init__(self, model_type="vit_h", device="cuda"):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sam = None
        self.mask_generator = None
        
        with tqdm(total=2, desc="Initializing SAM", leave=False) as pbar:
            pbar.set_description("Checking device")
            print(f"Using device: {self.device}")
            pbar.update(1)
            
            pbar.set_description("Loading model")
            self.initialize_model()
            pbar.update(1)

    def download_weights(self):
        model_info = SAM_MODELS[self.model_type]
        weight_path = SAM_WEIGHTS_DIR / model_info['checkpoint']
        
        if not weight_path.exists():
            with tqdm(total=1, desc="Checking weights", leave=False) as pbar:
                print(f"Weight file not found. Downloading {self.model_type} weights...")
                pbar.update(1)
            
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            try:
                with DownloadProgressBar(unit='B', unit_scale=True,
                                       miniters=1, desc=f"Downloading {self.model_type}",
                                       position=0, leave=True) as t:
                    urllib.request.urlretrieve(
                        model_info['url'],
                        weight_path,
                        reporthook=t.update_to,
                        context=ssl_context
                    )
            except Exception as e:
                print(f"Error downloading weights: {e}")
                try:
                    print("Trying alternative download method...")
                    ssl._create_default_https_context = ssl._create_unverified_context
                    with DownloadProgressBar(unit='B', unit_scale=True,
                                           miniters=1, desc="Downloading",
                                           position=0, leave=True) as t:
                        urllib.request.urlretrieve(
                            model_info['url'],
                            weight_path,
                            reporthook=t.update_to
                        )
                except Exception as e2:
                    raise Exception(f"Failed to download weights: {e2}")
        else:
            print(f"Using existing weights from: {weight_path}")
        
        return weight_path

    def initialize_model(self):
        with tqdm(total=4, desc="Model initialization", leave=False) as pbar:
            pbar.set_description("Downloading weights")
            weight_path = self.download_weights()
            pbar.update(1)
            
            pbar.set_description("Loading SAM model")
            self.sam = sam_model_registry[self.model_type](checkpoint=str(weight_path))
            pbar.update(1)
            
            pbar.set_description("Moving model to device")
            self.sam.to(self.device)
            pbar.update(1)
            
            pbar.set_description("Configuring mask generator")
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.7,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=50,
                output_mode="binary_mask"
            )
            pbar.update(1)

    def generate_masks(self, image):
        with tqdm(total=1, desc="Generating masks", leave=False) as pbar:
            masks = self.mask_generator.generate(image)
            pbar.update(1)
            print(f"Generated {len(masks)} masks")
            return masks 