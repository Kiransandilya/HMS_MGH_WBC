import os
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
    """Get input file path from user and validate, with default to a specific folder."""
    base_input_path = Path(r'F:\Kiran\Data\Input')  # Default folder if nothing is entered
    while True:
        input_path = input("Enter the path to TIFF file or folder (press Enter for default): ").strip()
        
        if not input_path:  # Edge Case 1: No input, use default folder
            print("No path entered. Searching in default folder...")
            selected_file = search_random_tiff(base_input_path)
            print(f"Randomly selected file: {selected_file}")
            return selected_file
        
        path = Path(input_path)

        if not path.exists():  # Invalid path
            print("Path does not exist. Please try again.")
            continue

        if path.is_file() and path.suffix.lower() in ['.tif', '.tiff']:  # Edge Case 3 (Exact file)
            print(f"Selected file: {path}")
            return path
        
        if path.is_dir():  # Edge Case 2 (Folder input)
            selected_file = search_random_tiff(path)
            print(f"Randomly selected file from directory: {selected_file}")
            return selected_file
        
        # Edge Case 3 (Partial filename), search in base directory
        # Look for files matching the input name (if it’s not a complete path)
        if path.is_file() == False and not path.suffix:  # This means it's likely a filename input
            print(f"Searching for file '{input_path}' in the default input folder...")
            possible_files = list(base_input_path.glob(f'**/{input_path}.tif')) + list(base_input_path.glob(f'**/{input_path}.tiff'))
            
            if not possible_files:
                print(f"No file named '{input_path}.tif' or '{input_path}.tiff' found.")
                continue
            
            if len(possible_files) == 1:  # Exactly one match
                print(f"Found the file: {possible_files[0]}")
                return possible_files[0]
            else:  # Multiple files, prompt the user to select
                print("Multiple files found with the same name:")
                for file in possible_files:
                    print(file)
                print("Please select one by entering the full file name or press Enter to randomly select.")
                continue  # Let the user try again

def search_random_tiff(folder_path):
    """Search the directory (including subdirectories) for a random TIFF file."""
    # Recursively find all .tif or .tiff files
    tiff_files = list(folder_path.glob('**/*.tif')) + list(folder_path.glob('**/*.tiff'))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {folder_path}")
    
    # Randomly select a file from the list
    selected_file = random.choice(tiff_files)
    return selected_file