import csv
import time
import psutil
import logging
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class ExperimentLogger:
    def __init__(self, log_dir, exp_log_path):
        self.log_dir = Path(log_dir)
        self.exp_log_path = Path(exp_log_path)
        self.start_time = None
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def get_next_log_number(self, base_name):
        existing_logs = list(self.log_dir.glob(f"{base_name}_*.log"))
        if not existing_logs:
            return 1
        numbers = [int(log.stem.split('_')[-1]) for log in existing_logs]
        return max(numbers) + 1

    def start_experiment(self):
        self.start_time = time.time()
        return self.start_time

    def create_detailed_log(self, input_file, output_path, model_name, frames_processed, 
                          processing_times, total_time):
        """Create a detailed log file for each input file processing."""
        with tqdm(total=3, desc="Creating detailed log", leave=False) as pbar:
            # Prepare log data
            detailed_log = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_file': str(input_file),
                'output_directory': str(output_path),
                'model_name': model_name,
                'total_frames_processed': frames_processed,
                'total_processing_time': f"{total_time:.2f} seconds",
                'average_time_per_frame': f"{(total_time/frames_processed):.2f} seconds",
                'frame_processing_times': {
                    f"frame_{i}": f"{time:.2f} seconds" 
                    for i, time in enumerate(processing_times)
                },
                'system_info': {
                    'cpu_usage': f"{psutil.cpu_percent()}%",
                    'memory_usage': f"{psutil.virtual_memory().percent}%",
                    'gpu_usage': f"{self.get_gpu_usage():.2f}%" if self.get_gpu_usage() > 0 else "N/A"
                }
            }
            pbar.update(1)

            # Create log filename
            base_name = input_file.stem
            log_number = self.get_next_log_number(base_name)
            log_file = self.log_dir / f"{base_name}_{log_number}.json"
            pbar.update(1)

            # Write detailed log
            with open(log_file, 'w') as f:
                json.dump(detailed_log, f, indent=4)
            pbar.update(1)

            print(f"Detailed log saved to: {log_file}")

    def log_experiment(self, input_file, output_path, model_name, frames_processed, 
                      processing_times, total_time):
        """Log experiment to CSV and create detailed log file."""
        with tqdm(total=2, desc="Logging experiment", leave=False) as pbar:
            # Log to CSV
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_file': str(input_file),
                'output_path': str(output_path),
                'model_name': model_name,
                'total_frames': frames_processed,
                'total_time': total_time,
                'avg_frame_time': sum(processing_times) / len(processing_times),
                'cpu_usage': psutil.cpu_percent(),
                'gpu_usage': self.get_gpu_usage()
            }
            self.write_to_csv(log_data)
            pbar.update(1)

            # Create detailed log file
            self.create_detailed_log(
                input_file, output_path, model_name,
                frames_processed, processing_times, total_time
            )
            pbar.update(1)

            return log_data

    def write_to_csv(self, log_data):
        file_exists = self.exp_log_path.exists()
        
        with open(self.exp_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

    @staticmethod
    def get_gpu_usage():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return 0
        except:
            return 0 