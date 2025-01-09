import cv2
import numpy as np
from collections import defaultdict

class BlobTracker:
    def __init__(self):
        self.tracks = defaultdict(list)
        self.next_id = 0

    def update_tracks(self, contours, frame_index):
        new_tracks = {}

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            matched_id = None
            min_distance = float('inf')

            for blob_id, positions in self.tracks.items():
                px, py, last_frame = positions[-1]
                if frame_index - last_frame > 2:
                    continue

                distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                if distance < min_distance and distance < 50:
                    matched_id = blob_id
                    min_distance = distance

            if matched_id is not None:
                new_tracks[matched_id] = (cx, cy, frame_index)
            else:
                new_id = self.next_id
                self.next_id += 1
                new_tracks[new_id] = (cx, cy, frame_index)

        for blob_id, (cx, cy, frame_index) in new_tracks.items():
            self.tracks[blob_id].append((cx, cy, frame_index))

    def get_tracks(self):
        return self.tracks
