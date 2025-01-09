import logging
from pathlib import Path
from datetime import datetime
import random

def setup_logging(log_folder):
    """Setup logging configuration"""
    log_folder = Path(log_folder)
    log_folder.mkdir(exist_ok=True)
    log_file = log_folder / f"cell_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_unique_output_dir(base_path, folder_name):
    """
    Create a unique output directory by appending numbers if the directory already exists.
    
    Args:
        base_path (Path): Base directory path
        folder_name (str): Desired folder name
    
    Returns:
        Path: Path object with unique folder name
    """
    counter = 0
    output_dir = base_path / folder_name
    
    while output_dir.exists():
        counter += 1
        new_name = f"{folder_name}_{counter}"
        output_dir = base_path / new_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_input_file():
    """Get input file path from user and validate"""
    while True:
        input_path = input("Enter the path to TIFF file or folder: ").strip()
        
        if not input_path:
            print("Please enter a valid path")
            continue
        
        path = Path(input_path)
        
        if not path.exists():
            print("Path does not exist")
            continue
        
        # Check if it's a file with .tif or .tiff extension
        if path.is_file() and path.suffix.lower() in ['.tif', '.tiff']:
            print(f"Selected file: {path}")
            return path
        
        # Check if it's a directory
        if path.is_dir():
            # Look for .tif and .tiff files
            tiff_files = list(path.glob('*.tif')) + list(path.glob('*.tiff'))
            if not tiff_files:
                print("No TIFF files found in directory")
                continue
            
            # Select a random file
            selected_file = random.choice(tiff_files)
            print(f"Randomly selected file from directory: {selected_file}")
            return selected_file
        
        print("Please enter a valid TIFF file or directory path")
