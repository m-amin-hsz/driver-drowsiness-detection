"""
Feature Extraction Module for Driver Drowsiness Detection.
Implements:
- EAR (Eye Aspect Ratio) for left and right eyes
- MAR (Mouth Aspect Ratio) for yawning detection
- PERCLOS (Percentage of Eye Closure over time window)
- Blink Rate & Blink Counter
- Head Tilt / Nodding indicator
"""

from collections import deque
import numpy as np


def euclidean_dist(p1, p2):
    """Calculate Euclidean distance between two 2D/3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio (EAR).
    eye_landmarks: list/array of 6 points: [p1, p2, p3, p4, p5, p6]
    p1: outer corner
    p2: top-left
    p3: top-right
    p4: inner corner
    p5: bottom-right
    p6: bottom-left
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_landmarks) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]

    # Vertical eye distances
    dist_vertical_1 = euclidean_dist(p2, p6)
    dist_vertical_2 = euclidean_dist(p3, p5)

    # Horizontal eye distance
    dist_horizontal = euclidean_dist(p1, p4)

    if dist_horizontal == 0:
        return 0.0

    ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
    return float(ear)


def calculate_mar(mouth_landmarks):
    """
    Compute Mouth Aspect Ratio (MAR) to detect yawning.
    mouth_landmarks: list/array containing:
    [left_corner, right_corner, top_lip, bottom_lip, optional top2, bottom2, top3, bottom3]
    Basic standard: (top - bottom) / (left - right)
    """
    if len(mouth_landmarks) < 4:
        return 0.0

    left = mouth_landmarks[0]
    right = mouth_landmarks[1]
    top = mouth_landmarks[2]
    bottom = mouth_landmarks[3]

    dist_horizontal = euclidean_dist(left, right)
    if dist_horizontal == 0:
        return 0.0

    if len(mouth_landmarks) >= 8:
        # Multi-point MAR for higher accuracy
        top2, bottom2 = mouth_landmarks[4], mouth_landmarks[5]
        top3, bottom3 = mouth_landmarks[6], mouth_landmarks[7]
        v1 = euclidean_dist(top, bottom)
        v2 = euclidean_dist(top2, bottom2)
        v3 = euclidean_dist(top3, bottom3)
        mar = (v1 + v2 + v3) / (2.0 * dist_horizontal)
    else:
        v = euclidean_dist(top, bottom)
        mar = v / dist_horizontal

    return float(mar)


class TemporalDrowsinessTracker:
    """
    Maintains temporal metrics over sliding window of frames:
    - PERCLOS (Percentage of Eye Closure)
    - Blink Rate (blinks per minute or blinks in current window)
    - Average EAR and MAR over window
    """

    def __init__(self, window_size=90, ear_closed_thresh=0.21, fps=30):
        """
        window_size: Number of frames in sliding window (e.g. 90 frames = 3 seconds at 30 fps)
        ear_closed_thresh: EAR threshold below which eye is considered closed
        fps: Expected frames per second
        """
        self.window_size = window_size
        self.ear_closed_thresh = ear_closed_thresh
        self.fps = fps

        self.ear_history = deque(maxlen=window_size)
        self.mar_history = deque(maxlen=window_size)
        self.closed_history = deque(maxlen=window_size)

        # Blink counting state
        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.blink_timestamps = deque()  # stores frame indices of blinks
        self.frame_idx = 0

    def update(self, ear_avg, mar):
        """
        Update tracker with current frame's EAR and MAR.
        Returns a dictionary of computed features.
        """
        self.frame_idx += 1
        is_closed = 1 if ear_avg < self.ear_closed_thresh else 0

        self.ear_history.append(ear_avg)
        self.mar_history.append(mar)
        self.closed_history.append(is_closed)

        # Blink detection logic: eye closes for >= 2 frames and <= 15 frames (micro-sleep vs normal blink)
        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            if 2 <= self.consecutive_closed_frames <= 15:
                self.total_blinks += 1
                self.blink_timestamps.append(self.frame_idx)
            self.consecutive_closed_frames = 0

        # Purge blinks older than current window
        cutoff_frame = self.frame_idx - self.window_size
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff_frame:
            self.blink_timestamps.popleft()

        # Calculate PERCLOS: percentage of frames eyes were closed in window
        if len(self.closed_history) > 0:
            perclos = (sum(self.closed_history) / len(self.closed_history)) * 100.0
        else:
            perclos = 0.0

        # Calculate Blinks per minute (normalized by window length)
        window_duration_seconds = len(self.closed_history) / max(1.0, float(self.fps))
        if window_duration_seconds > 0:
            blinks_per_minute = (len(self.blink_timestamps) / window_duration_seconds) * 60.0
        else:
            blinks_per_minute = 0.0

        return {
            "ear": float(ear_avg),
            "mar": float(mar),
            "perclos": float(perclos),
            "blinks_in_window": len(self.blink_timestamps),
            "blink_rate_bpm": float(blinks_per_minute),
            "ear_rolling_mean": float(np.mean(self.ear_history)),
            "mar_rolling_mean": float(np.mean(self.mar_history)),
            "consecutive_closed_frames": self.consecutive_closed_frames
        }

    def reset(self):
        """Reset internal history buffers."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.closed_history.clear()
        self.blink_timestamps.clear()
        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.frame_idx = 0

