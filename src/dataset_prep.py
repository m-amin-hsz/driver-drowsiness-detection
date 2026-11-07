"""
Dataset Preparation, Cleaning, and Train/Val/Test Splitting Module.
Ensures Person 1 provides a clean, well-partitioned dataset manifest for Person 2.
"""

import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}


def is_valid_file(file_path: str) -> bool:
    """Check if the image/video file is non-empty and readable."""
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False

    if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(file_path)
        return img is not None and img.shape[0] > 10 and img.shape[1] > 10
    elif ext in {".mp4", ".avi", ".mov"}:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    return True


def scan_and_index_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Traverse subdirectories in raw_dir and build metadata DataFrame.
    Assumes structure:
      raw_dir/
        ├── awake/ (or non-drowsy/normal/alert)
        └── drowsy/ (or yawning/sleepy/closed)
    """
    records = []
    awake_keywords = {"awake", "open", "normal", "non-drowsy", "alert"}
    drowsy_keywords = {"drowsy", "closed", "yawn", "yawning", "sleepy", "nodding"}

    for root, _, files in os.walk(raw_dir):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            folder_name = os.path.basename(root).lower()
            file_name = f.lower()

            # Determine label
            label = None
            if any(k in folder_name or k in file_name for k in drowsy_keywords):
                label = "drowsy"
            elif any(k in folder_name or k in file_name for k in awake_keywords):
                label = "awake"
            else:
                label = folder_name  # fallback to parent directory name

            if is_valid_file(path):
                is_video = ext in {".mp4", ".avi", ".mov"}
                records.append({
                    "file_path": os.path.abspath(path),
                    "file_name": f,
                    "media_type": "video" if is_video else "image",
                    "label": label,
                    "target": 1 if label == "drowsy" else 0
                })
            else:
                print(f"⚠️ Warning: Skipping corrupted or unreadable file: {path}")

    return pd.DataFrame(records)


def create_splits(
    df: pd.DataFrame,
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split dataset into Stratified Train / Validation / Test sets and save to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(df) == 0:
        print("⚠️ Warning: No valid files found to split.")
        return

    # Check if we have enough samples per class for stratification
    min_class_count = df["label"].value_counts().min() if df["label"].nunique() > 1 else 0
    stratify_col = df["label"] if min_class_count >= 2 else None

    # First split: Train vs (Val + Test)
    test_val_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=test_val_ratio,
        random_state=seed,
        stratify=stratify_col
    )

    # Second split: Val vs Test
    temp_min_count = temp_df["label"].value_counts().min() if temp_df["label"].nunique() > 1 else 0
    temp_stratify = temp_df["label"] if temp_min_count >= 2 else None
    relative_test_ratio = test_ratio / test_val_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_stratify
    )

    # Save to CSV
    train_path = os.path.join(output_dir, "train_manifest.csv")
    val_path = os.path.join(output_dir, "val_manifest.csv")
    test_path = os.path.join(output_dir, "test_manifest.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print("✅ DATASET SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total samples:      {len(df)}")
    print(f"Train samples (70%): {len(train_df)} -> {train_path}")
    print(f"Val samples   (15%): {len(val_df)}   -> {val_path}")
    print(f"Test samples  (15%): {len(test_df)}  -> {test_path}")
    print("\nClass Distribution in Train:")
    print(train_df["label"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw data and generate Train/Val/Test splits.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw datasets folder")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Output directory for manifests")
    args = parser.parse_args()

    df_dataset = scan_and_index_dataset(args.raw_dir)
    if len(df_dataset) > 0:
        create_splits(df_dataset, output_dir=args.out_dir)
    else:
        print(f"No files found in {args.raw_dir}. Try running src/dataset_downloader.py first!")
