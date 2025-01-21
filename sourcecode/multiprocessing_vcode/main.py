import time
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from .sam_processor import SAMProcessor
from .image_utils import ImageProcessor
from .logging_utils import ExperimentLogger
from .visualization import Visualizer
from .config import *

def process_frames_chunk(chunk_data):
    """Process a chunk of frames on a specific GPU."""
    frames, device_id, model_type, output_dir = chunk_data
    
    # Initialize processor for this GPU
    sam_processor = SAMProcessor(model_type=model_type, device_id=device_id)
    img_processor = ImageProcessor()
    processing_times = []
    
    # Process frames in this chunk
    with tqdm(total=len(frames), desc=f"GPU {device_id} Progress", position=device_id) as pbar:
        for idx, frame in enumerate(frames):
            frame_start = time.time()
            
            # Process frame
            original, rgb_frame = img_processor.preprocess_frame(frame)
            masks = sam_processor.generate_masks(rgb_frame)
            filtered_masks = img_processor.filter_masks(masks, original)
            
            # Save output
            output_path = output_dir / f"frame_{idx:04d}_gpu{device_id}.png"
            Visualizer.visualize_masks(original, filtered_masks, output_path)
            
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            pbar.set_postfix({'Time': f'{frame_time:.2f}s'})
            pbar.update(1)
    
    return processing_times

def process_tiff(input_file, frame_range=None, model_type="vit_h"):
    print(f"\nInitializing processing for: {input_file.name}")
    
    # Get available GPUs
    gpu_info = SAMProcessor.get_available_gpus()
    num_gpus = len(gpu_info)
    
    if num_gpus == 0:
        print("No GPUs available, processing on CPU")
        num_gpus = 1
    else:
        print(f"Found {num_gpus} GPUs:")
        for gpu in gpu_info:
            print(f"GPU {gpu['id']}: {gpu['name']}")
    
    # Create output directory
    base_name = input_file.stem
    output_dir = SAM_OUTPUTS_DIR / base_name
    next_num = 1
    while (output_dir / f"{next_num}").exists():
        next_num += 1
    output_dir = output_dir / f"{next_num}"
    output_dir.mkdir(parents=True)
    
    # Initialize logger
    logger = ExperimentLogger(SAM_LOGS_DIR, EXPERIMENTAL_LOGS_DIR / "experiments.csv")
    start_time = logger.start_experiment()
    
    # Read frames
    img_processor = ImageProcessor()
    frames = img_processor.read_tiff(input_file)
    
    if frame_range:
        start, end = frame_range
        frames = frames[start:end]
        print(f"Processing frames {start} to {end}")
    else:
        print(f"Processing all {len(frames)} frames")
    
    # Split frames among available GPUs
    frames_per_gpu = len(frames) // num_gpus
    frame_chunks = []
    
    for i in range(num_gpus):
        start_idx = i * frames_per_gpu
        end_idx = start_idx + frames_per_gpu if i < num_gpus - 1 else len(frames)
        chunk = (
            frames[start_idx:end_idx],
            i % num_gpus,  # GPU device ID
            model_type,
            output_dir
        )
        frame_chunks.append(chunk)
    
    # Process chunks in parallel
    if num_gpus > 1:
        mp.set_start_method('spawn', force=True)
        with mp.Pool(num_gpus) as pool:
            results = pool.map(process_frames_chunk, frame_chunks)
        processing_times = [time for chunk_times in results for time in chunk_times]
    else:
        processing_times = process_frames_chunk(frame_chunks[0])
    
    total_time = time.time() - start_time
    
    # Log experiment
    logger.log_experiment(
        input_file=input_file,
        output_path=output_dir,
        model_name=model_type,
        frames_processed=len(frames),
        processing_times=processing_times,
        total_time=total_time
    )

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