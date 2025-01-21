import time
from pathlib import Path
from tqdm import tqdm
from .sam_processor import SAMProcessor
from .image_utils import ImageProcessor
from .logging_utils import ExperimentLogger
from .visualization import Visualizer
from .config import *

def process_tiff(input_file, frame_range=None, model_type="vit_h"):
    print(f"\nInitializing processing for: {input_file.name}")
    
    # Initialize processors and logger with progress bar
    with tqdm(total=3, desc="Setup", leave=False) as pbar:
        sam_processor = SAMProcessor(model_type=model_type)
        pbar.update(1)
        
        img_processor = ImageProcessor()
        pbar.update(1)
        
        # Create output directory
        base_name = input_file.stem
        output_dir = SAM_OUTPUTS_DIR / base_name
        next_num = 1
        while (output_dir / f"{next_num}").exists():
            next_num += 1
        output_dir = output_dir / f"{next_num}"
        output_dir.mkdir(parents=True)
        pbar.update(1)
    
    # Initialize logger
    logger = ExperimentLogger(SAM_LOGS_DIR, EXPERIMENTAL_LOGS_DIR / "experiments.csv")
    start_time = logger.start_experiment()
    
    # Read frames with progress bar
    with tqdm(desc="Reading TIFF file", leave=False) as pbar:
        frames = img_processor.read_tiff(input_file)
        pbar.update(1)
    
    if frame_range:
        start, end = frame_range
        frames = frames[start:end]
        print(f"Processing frames {start} to {end}")
    else:
        print(f"Processing all {len(frames)} frames")
    
    processing_times = []
    
    # Main processing loop with nested progress bars
    with tqdm(total=len(frames), desc="Overall Progress", position=0) as pbar_main:
        for idx, frame in enumerate(frames):
            frame_start = time.time()
            
            # Process frame with nested progress bars
            with tqdm(total=4, desc=f"Frame {idx}", position=1, leave=False) as pbar_frame:
                # Preprocess
                pbar_frame.set_description(f"Frame {idx}: Preprocessing")
                original, rgb_frame = img_processor.preprocess_frame(frame)
                pbar_frame.update(1)
                
                # Generate masks
                pbar_frame.set_description(f"Frame {idx}: Generating masks")
                masks = sam_processor.generate_masks(rgb_frame)
                pbar_frame.update(1)
                
                # Filter masks
                pbar_frame.set_description(f"Frame {idx}: Filtering masks")
                filtered_masks = img_processor.filter_masks(masks, original)
                pbar_frame.update(1)
                
                # Save output
                pbar_frame.set_description(f"Frame {idx}: Saving output")
                output_path = output_dir / f"frame_{idx:04d}.png"
                Visualizer.visualize_masks(original, filtered_masks, output_path)
                pbar_frame.update(1)
            
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            # Update main progress bar with timing information
            avg_time = sum(processing_times) / len(processing_times)
            pbar_main.set_postfix({
                'Current': f'{frame_time:.2f}s',
                'Avg': f'{avg_time:.2f}s'
            })
            pbar_main.update(1)
    
    total_time = time.time() - start_time
    
    # Log experiment with progress bar
    with tqdm(total=1, desc="Logging experiment", leave=False) as pbar:
        logger.log_experiment(
            input_file=input_file,
            output_path=output_dir,
            model_name=model_type,
            frames_processed=len(frames),
            processing_times=processing_times,
            total_time=total_time
        )
        pbar.update(1)

    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Average time per frame: {sum(processing_times)/len(processing_times):.2f} seconds")
    print(f"Results saved to: {output_dir}")

def main():
    # Process all TIFF files in input directory with progress bar
    tiff_files = list(INPUT_DIR.glob("*.tiff"))
    if not tiff_files:
        print("No TIFF files found in input directory")
        return
        
    print(f"Found {len(tiff_files)} TIFF files to process")
    for tiff_file in tqdm(tiff_files, desc="Processing files"):
        process_tiff(tiff_file)

if __name__ == "__main__":
    main() 