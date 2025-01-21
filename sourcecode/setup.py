from pathlib import Path
from .config import *  # If you need config variables

def create_directory_structure():
    # Define base directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "Data"
    
    # Define all required directories
    directories = [
        data_dir / "Input",
        data_dir / "ExperimentalLogs",
        data_dir / "Sam" / "samweights",
        data_dir / "Sam" / "samcheckpoints",
        data_dir / "Sam" / "samoutputs",
        data_dir / "Sam" / "samlogs",
    ]
    
    # Create all directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directory_structure() 