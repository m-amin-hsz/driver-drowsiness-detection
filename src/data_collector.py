"""
Webcam Data Collector for Driver Drowsiness Detection.
Allows Person 1 to easily record custom video clips or frames of 'awake' and 'drowsy' states.
Keys:
  'a' : Start/Stop recording AWAKE video
  'd' : Start/Stop recording DROWSY video
  's' : Snapshot current frame
  'q' : Quit
"""

import os
import time
import argparse
import cv2


def run_collector(output_dir: str = "data/raw/custom", camera_id: int = 0):
    awake_dir = os.path.join(output_dir, "awake")
    drowsy_dir = os.path.join(output_dir, "drowsy")
    os.makedirs(awake_dir, exist_ok=True)
    os.makedirs(drowsy_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}.")
        print("Tip: If you don't have a webcam connected, you can supply video files to src/pipeline.py directly.")
        return

    current_label = None
    video_writer = None
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print("=" * 60)
    print("🎥 DRIVER DROWSINESS DATA COLLECTOR")
    print("=" * 60)
    print("Controls:")
    print("  [A] - Toggle recording AWAKE clip")
    print("  [D] - Toggle recording DROWSY clip")
    print("  [S] - Save single snapshot")
    print("  [Q] - Quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            display = frame.copy()
            status_text = f"Recording: {current_label.upper()}" if current_label else "Status: IDLE"
            status_color = (0, 0, 255) if current_label == "drowsy" else ((0, 255, 0) if current_label == "awake" else (200, 200, 200))

            cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display, "Press 'a': Awake | 'd': Drowsy | 'q': Quit", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if video_writer is not None:
                video_writer.write(frame)

            try:
                cv2.imshow("Data Collector", display)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                # Running headless without DISPLAY
                key = 0xFF

            if key == ord('q'):
                break
            elif key == ord('a'):
                if current_label == "awake":
                    # Stop
                    print("⏹️ Stopped recording AWAKE.")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    current_label = None
                else:
                    if video_writer:
                        video_writer.release()
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(awake_dir, f"awake_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    current_label = "awake"
                    print(f"▶️ Recording AWAKE to {out_path}...")
            elif key == ord('d'):
                if current_label == "drowsy":
                    # Stop
                    print("⏹️ Stopped recording DROWSY.")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    current_label = None
                else:
                    if video_writer:
                        video_writer.release()
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(drowsy_dir, f"drowsy_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    current_label = "drowsy"
                    print(f"▶️ Recording DROWSY to {out_path}...")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                target = awake_dir if current_label == "awake" else drowsy_dir
                snap_path = os.path.join(target, f"snap_{timestamp}.jpg")
                cv2.imwrite(snap_path, frame)
                print(f"📸 Snapshot saved: {snap_path}")

    finally:
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record custom drowsiness dataset via camera.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--out", type=str, default="data/raw/custom", help="Output directory")
    args = parser.parse_args()
    run_collector(output_dir=args.out, camera_id=args.camera)

