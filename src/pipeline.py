"""
Feature Extraction Pipeline for Driver Drowsiness Detection.
Transforms raw video or image frames into a structured tabular feature dataset
ready for Machine Learning modeling (Person 2's role).
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import pandas as pd
from tqdm import tqdm

from src.face_mesh import FaceMeshDetector
from src.feature_extraction import calculate_ear, calculate_mar, TemporalDrowsinessTracker


def process_image(detector: FaceMeshDetector, img_bgr, label: str = "unknown") -> dict:
    """Extract full feature row from a single image."""
    result = detector.process_frame(img_bgr)
    if result is None:
        return None

    ear_r = calculate_ear(result["right_eye_pts"])
    ear_l = calculate_ear(result["left_eye_pts"])
    ear_avg = (ear_r + ear_l) / 2.0
    mar = calculate_mar(result["mouth_pts"])

    blendshapes = result.get("blendshapes", {})

    return {
        "ear_left": round(ear_l, 4),
        "ear_right": round(ear_r, 4),
        "ear_avg": round(ear_avg, 4),
        "mar": round(mar, 4),
        "pitch_ratio": round(result["pitch_ratio"], 4),
        "blendshape_blink_l": round(blendshapes.get("eyeBlinkLeft", 0.0), 4),
        "blendshape_blink_r": round(blendshapes.get("eyeBlinkRight", 0.0), 4),
        "blendshape_jaw_open": round(blendshapes.get("jawOpen", 0.0), 4),
        "label": label,
        "target": 1 if label.lower() in {"drowsy", "closed", "yawn", "sleepy"} else 0
    }


def process_video(
    detector: FaceMeshDetector,
    video_path: str,
    label: str = "unknown",
    sample_every_n_frames: int = 2
) -> list:
    """Extract temporal feature sequence from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracker = TemporalDrowsinessTracker(window_size=int(fps * 3), ear_closed_thresh=0.21, fps=fps)
    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n_frames != 0:
            continue

        result = detector.process_frame(frame)
        if result is None:
            continue

        ear_r = calculate_ear(result["right_eye_pts"])
        ear_l = calculate_ear(result["left_eye_pts"])
        ear_avg = (ear_r + ear_l) / 2.0
        mar = calculate_mar(result["mouth_pts"])

        temporal = tracker.update(ear_avg, mar)
        blendshapes = result.get("blendshapes", {})

        row = {
            "source_file": os.path.basename(video_path),
            "frame_idx": frame_idx,
            "ear_left": round(ear_l, 4),
            "ear_right": round(ear_r, 4),
            "ear_avg": round(ear_avg, 4),
            "mar": round(mar, 4),
            "perclos": round(temporal["perclos"], 2),
            "blinks_in_window": temporal["blinks_in_window"],
            "blink_rate_bpm": round(temporal["blink_rate_bpm"], 1),
            "consecutive_closed_frames": temporal["consecutive_closed_frames"],
            "pitch_ratio": round(result["pitch_ratio"], 4),
            "blendshape_blink_l": round(blendshapes.get("eyeBlinkLeft", 0.0), 4),
            "blendshape_blink_r": round(blendshapes.get("eyeBlinkRight", 0.0), 4),
            "blendshape_jaw_open": round(blendshapes.get("jawOpen", 0.0), 4),
            "label": label,
            "target": 1 if label.lower() in {"drowsy", "closed", "yawn", "sleepy"} else 0
        }
        rows.append(row)

    cap.release()
    return rows


def build_feature_dataset(
    manifest_csv: str,
    output_csv: str,
    detector: FaceMeshDetector
):
    """
    Process all items in a manifest CSV and save tabular features CSV.
    """
    df_manifest = pd.read_csv(manifest_csv)
    all_features = []

    print(f"Processing {len(df_manifest)} entries from manifest: {manifest_csv}")
    for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest)):
        file_path = row["file_path"]
        label = row["label"]
        media_type = row["media_type"]

        if media_type == "video":
            video_rows = process_video(detector, file_path, label=label)
            all_features.extend(video_rows)
        else:
            img = cv2.imread(file_path)
            if img is not None:
                feat = process_image(detector, img, label=label)
                if feat is not None:
                    feat["source_file"] = os.path.basename(file_path)
                    all_features.append(feat)

    if not all_features:
        print("⚠️ No features could be extracted (no faces detected in manifest files).")
        return None

    df_out = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"🎉 Features dataset successfully saved: {output_csv} ({len(df_out)} rows)")
    return df_out


def run_live_demo(detector: FaceMeshDetector, camera_id: int = 0):
    """Run interactive real-time visual demonstration."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}.")
        return

    tracker = TemporalDrowsinessTracker(window_size=90, ear_closed_thresh=0.21, fps=30)
    print("Running live demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        if result is not None:
            ear_r = calculate_ear(result["right_eye_pts"])
            ear_l = calculate_ear(result["left_eye_pts"])
            ear_avg = (ear_r + ear_l) / 2.0
            mar = calculate_mar(result["mouth_pts"])

            metrics = tracker.update(ear_avg, mar)
            frame = detector.draw_landmarks(frame, result)

            # Heuristic quick alert for display
            is_drowsy = (metrics["perclos"] > 35.0) or (metrics["consecutive_closed_frames"] > 20) or (mar > 0.65)
            status_text = "DROWSY ⚠️" if is_drowsy else "AWAKE ✅"
            color = (0, 0, 255) if is_drowsy else (0, 255, 0)

            # HUD Display
            cv2.putText(frame, f"STATUS: {status_text}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"EAR: {ear_avg:.2f} (L:{ear_l:.2f}, R:{ear_r:.2f})", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"PERCLOS: {metrics['perclos']:.1f}%", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Blink Rate: {metrics['blink_rate_bpm']:.1f} bpm", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        try:
            cv2.imshow("Driver Drowsiness - Feature Extraction Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Headless environment
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Feature Extraction Pipeline.")
    parser.add_argument("--manifest", type=str, default="", help="Path to manifest CSV from dataset_prep")
    parser.add_argument("--out", type=str, default="data/features/features.csv", help="Output CSV path")
    parser.add_argument("--live", action="store_true", help="Launch live webcam visualization")
    args = parser.parse_args()

    detector = FaceMeshDetector()

    if args.live:
        run_live_demo(detector)
    elif args.manifest:
        build_feature_dataset(args.manifest, args.out, detector)
    else:
        print("Please provide --manifest path/to/manifest.csv or --live")
