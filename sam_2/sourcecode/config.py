import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"

# Input/Output directories
INPUT_DIR = DATA_DIR / "Input"
SAM_DIR = DATA_DIR / "Sam"
EXPERIMENTAL_LOGS_DIR = DATA_DIR / "ExperimentalLogs"

# SAM subdirectories
SAM_WEIGHTS_DIR = SAM_DIR / "samweights"
SAM_CHECKPOINTS_DIR = SAM_DIR / "samcheckpoints"
SAM_OUTPUTS_DIR = SAM_DIR / "samoutputs"
SAM_LOGS_DIR = SAM_DIR / "samlogs"

# Create all directories
for dir_path in [INPUT_DIR, SAM_DIR, EXPERIMENTAL_LOGS_DIR, 
                SAM_WEIGHTS_DIR, SAM_CHECKPOINTS_DIR, 
                SAM_OUTPUTS_DIR, SAM_LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# SAM model configurations
SAM_MODELS = {
    'vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'checkpoint': 'sam_vit_h_4b8939.pth'
    },
    'vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'checkpoint': 'sam_vit_l_0b3195.pth'
    },
    'vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'checkpoint': 'sam_vit_b_01ec64.pth'
    }
} 