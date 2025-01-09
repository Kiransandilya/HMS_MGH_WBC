import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os

class CellTrackingMetrics:
    def __init__(self):
        self.metrics = defaultdict(dict)

    def calculate_metrics(self, track_positions, frame_count):
        if len(track_positions) < 2:
            return None

        positions = [(x, y, t) for x, y, t in track_positions]
        velocities = []
        accelerations = []
        angles = []
        timestamps = []
        x_coords = []
        y_coords = []

        for i in range(1, len(positions)):
            dt = positions[i][2] - positions[i - 1][2]
            if dt == 0:
                continue

            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]

            x_coords.append(positions[i][0])
            y_coords.append(positions[i][1])

            distance = np.sqrt(dx ** 2 + dy ** 2)
            velocity = distance / dt
            velocities.append(velocity)
            timestamps.append(positions[i][2])

            angle = np.degrees(np.arctan2(dy, dx))
            angles.append(angle)

            if i > 1:
                prev_velocity = velocities[-2]
                acceleration = (velocity - prev_velocity) / dt
                accelerations.append(acceleration)

        accelerations = [0] + accelerations

        metrics = {
            'total_distance': sum(velocities),
            'displacement': np.sqrt(
                (positions[-1][0] - positions[0][0]) ** 2 +
                (positions[-1][1] - positions[0][1]) ** 2
            ),
            'avg_velocity': np.mean(velocities),
            'max_velocity': max(velocities),
            'min_velocity': min(velocities),
            'velocity_std': np.std(velocities),
            'avg_acceleration': np.mean(accelerations),
            'directional_changes': self._count_direction_changes(angles),
            'confinement_ratio': self._calculate_confinement_ratio(track_positions),
            'mean_square_displacement': self._calculate_msd(track_positions),
            'track_duration': positions[-1][2] - positions[0][2],
            'movement_data': {
                'timestamps': timestamps,
                'velocities': velocities,
                'accelerations': accelerations,
                'x_coords': x_coords,
                'y_coords': y_coords
            }
        }

        return metrics

    def _count_direction_changes(self, angles, threshold=45):
        if len(angles) < 2:
            return 0
        direction_changes = 0
        for i in range(1, len(angles)):
            angle_diff = abs(angles[i] - angles[i - 1])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > threshold:
                direction_changes += 1
        return direction_changes

    def _calculate_confinement_ratio(self, positions):
        if len(positions) < 2:
            return 0
        end_to_end = np.sqrt(
            (positions[-1][0] - positions[0][0]) ** 2 +
            (positions[-1][1] - positions[0][1]) ** 2
        )
        total_length = sum(
            np.sqrt((positions[i + 1][0] - positions[i][0]) ** 2 +
                    (positions[i + 1][1] - positions[i][1]) ** 2)
            for i in range(len(positions) - 1)
        )
        return end_to_end / total_length if total_length > 0 else 0

    def _calculate_msd(self, positions):
        if len(positions) < 2:
            return 0
        displacements = []
        for tau in range(1, min(11, len(positions))):
            squared_displacements = []
            for i in range(len(positions) - tau):
                d_x = positions[i + tau][0] - positions[i][0]
                d_y = positions[i + tau][1] - positions[i][1]
                squared_displacements.append(d_x ** 2 + d_y ** 2)
            displacements.append(np.mean(squared_displacements))
        return np.mean(displacements)

    def plot_movement_graphs(self, cell_id, metrics, output_dir):
        """Plot movement analysis graphs and save data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        try:
            # Create figure
            plt.ioff()  # Turn off interactive mode
            fig = plt.figure(figsize=(20, 15))

            # Cell trajectory plot
            ax1 = fig.add_subplot(221)
            ax1.plot(metrics['movement_data']['x_coords'],
                     metrics['movement_data']['y_coords'],
                     'b-', linewidth=2)
            ax1.scatter(metrics['movement_data']['x_coords'][0],
                        metrics['movement_data']['y_coords'][0],
                        color='green', s=100, label='Start')
            ax1.scatter(metrics['movement_data']['x_coords'][-1],
                        metrics['movement_data']['y_coords'][-1],
                        color='red', s=100, label='End')
            ax1.set_title(f'Cell {cell_id} Trajectory')
            ax1.set_xlabel('X Position (pixels)')
            ax1.set_ylabel('Y Position (pixels)')
            ax1.legend()
            ax1.grid(True)

            # Velocity vs Time plot
            ax2 = fig.add_subplot(222)
            ax2.plot(metrics['movement_data']['timestamps'],
                     metrics['movement_data']['velocities'],
                     'g-', linewidth=2)
            ax2.set_title(f'Cell {cell_id} Velocity vs Time')
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Velocity (pixels/frame)')
            ax2.grid(True)

            # Acceleration vs Time plot
            ax3 = fig.add_subplot(223)
            ax3.plot(metrics['movement_data']['timestamps'],
                     metrics['movement_data']['accelerations'],
                     'r-', linewidth=2)
            ax3.set_title(f'Cell {cell_id} Acceleration vs Time')
            ax3.set_xlabel('Frame Number')
            ax3.set_ylabel('Acceleration (pixels/frame²)')
            ax3.grid(True)

            # Movement pattern analysis
            ax4 = fig.add_subplot(224)
            ax4.plot(metrics['movement_data']['timestamps'],
                     metrics['movement_data']['x_coords'],
                     'b-', label='X Position', linewidth=2)
            ax4.plot(metrics['movement_data']['timestamps'],
                     metrics['movement_data']['y_coords'],
                     'r-', label='Y Position', linewidth=2)
            ax4.set_title(f'Cell {cell_id} Position vs Time')
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('Position (pixels)')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            
            # Save the figure
            output_path = output_dir / f'cell_{cell_id}_movement_analysis.png'
            fig.savefig(str(output_path), dpi=300, bbox_inches='tight', format='png')
            plt.close(fig)

            # Save movement data to CSV
            movement_df = pd.DataFrame({
                'frame': metrics['movement_data']['timestamps'],
                'x_position': metrics['movement_data']['x_coords'],
                'y_position': metrics['movement_data']['y_coords'],
                'velocity': metrics['movement_data']['velocities'],
                'acceleration': metrics['movement_data']['accelerations']
            })
            csv_path = output_dir / f'cell_{cell_id}_movement_data.csv'
            movement_df.to_csv(str(csv_path), index=False)

        except Exception as e:
            print(f"Error saving plots: {str(e)}")
            plt.close('all')  # Clean up any open figures
        finally:
            plt.ion()  # Turn interactive mode back on
