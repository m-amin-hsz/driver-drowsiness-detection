"""
Face and Facial Landmark Detection Module using MediaPipe FaceLandmarker.
Provides unified extraction of face landmarks, eye points, mouth points, and blendshapes.
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canonical MediaPipe landmark indices for Eyes and Mouth
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 13, 14, 82, 87, 312, 317]
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10


class FaceMeshDetector:
    """
    Detector class wrapping MediaPipe FaceLandmarker tasks API.
    Works seamlessly on single frames, video streams, and webcam feeds.
    """

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not os.path.isabs(model_path):
            # Resolve relative to project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single BGR image/frame and extract coordinates and facial features.
        Returns None if no face is detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        # Take primary face (index 0)
        landmarks = result.face_landmarks[0]

        # Convert normalized coordinates (0..1) to pixel coordinates (x, y)
        coords = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])

        # Extract eye points
        right_eye_pts = coords[RIGHT_EYE_INDICES]
        left_eye_pts = coords[LEFT_EYE_INDICES]

        # Extract mouth points
        mouth_pts = coords[MOUTH_INDICES]

        # Head tilt proxy (ratio of vertical nose-chin to forehead-nose)
        nose_pt = coords[NOSE_TIP_INDEX]
        chin_pt = coords[CHIN_INDEX]
        forehead_pt = coords[FOREHEAD_INDEX]

        head_vertical_span = np.linalg.norm(chin_pt[:2] - forehead_pt[:2])
        nose_to_chin = np.linalg.norm(chin_pt[:2] - nose_pt[:2])
        pitch_ratio = float(nose_to_chin / head_vertical_span) if head_vertical_span > 0 else 0.5

        # Extract blendshapes if available
        blendshapes_dict = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for category in result.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = category.score

        # Approximate face bounding box
        x_min, y_min = np.min(coords[:, :2], axis=0)
        x_max, y_max = np.max(coords[:, :2], axis=0)
        bbox = (int(max(0, x_min)), int(max(0, y_min)), int(min(w, x_max)), int(min(h, y_max)))

        return {
            "bbox": bbox,
            "right_eye_pts": right_eye_pts,
            "left_eye_pts": left_eye_pts,
            "mouth_pts": mouth_pts,
            "all_landmarks": coords,
            "pitch_ratio": pitch_ratio,
            "blendshapes": blendshapes_dict,
        }

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        detection_data: Dict[str, Any],
        draw_mesh: bool = False,
    ) -> np.ndarray:
        """
        Draw visual annotations (eyes, mouth, bounding box) on frame for debugging.
        """
        output = frame_bgr.copy()
        if not detection_data:
            return output

        # Draw Face Bounding Box
        x1, y1, x2, y2 = detection_data["bbox"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Left Eye (Cyan)
        left_pts = detection_data["left_eye_pts"][:, :2].astype(int)
        for pt in left_pts:
            cv2.circle(output, tuple(pt), 2, (255, 255, 0), -1)
        cv2.polylines(output, [left_pts], isClosed=True, color=(255, 255, 0), thickness=1)

        # Draw Right Eye (Yellow)
        right_pts = detection_data["right_eye_pts"][:, :2].astype(int)
        for pt in right_pts:
            cv2.circle(output, tuple(pt), 2, (0, 255, 255), -1)
        cv2.polylines(output, [right_pts], isClosed=True, color=(0, 255, 255), thickness=1)

        # Draw Mouth (Magenta)
        mouth_pts = detection_data["mouth_pts"][:, :2].astype(int)
        for pt in mouth_pts:
            cv2.circle(output, tuple(pt), 2, (255, 0, 255), -1)

        return output
