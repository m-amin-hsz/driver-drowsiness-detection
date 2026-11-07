"""
Dataset Downloader & Reference Guide for Driver Drowsiness Detection.
Helps Person 1 acquire and set up benchmark datasets.
"""

import os
import sys
import json
import urllib.request
import zipfile
import shutil

# Benchmark datasets catalog
DATASET_CATALOG = {
    "mrl_eyes": {
        "name": "MRL Eye Dataset",
        "description": "Large-scale dataset of human eyes (>80,000 images) under various illumination, glasses, and reflections.",
        "url": "http://mrl.cs.vsb.cz/eyedataset",
        "type": "eye_crops",
        "classes": ["open", "closed"],
    },
    "cew": {
        "name": "Closed Eyes in the Wild (CEW)",
        "description": "2,423 subjects with annotated open/closed eye images taken in real-world scenarios.",
        "url": "http://parnec.nuaa.edu.cn/xt/download.html",
        "type": "eye_crops",
        "classes": ["open", "closed"],
    },
    "kaggle_ddd": {
        "name": "Driver Drowsiness Dataset (DDD)",
        "description": "Face-level images of drivers classified as Drowsy vs Non-Drowsy.",
        "url": "https://www.kaggle.com/datasets/isojack/drowsiness-dataset",
        "type": "face_images",
        "classes": ["Drowsy", "Non-Drowsy"],
    },
    "nthu_ddd": {
        "name": "NTHU Drowsy Driver Detection Dataset",
        "description": "High quality video dataset recorded in simulated driving scenarios with day/night, glasses/no-glasses, yawning, nodding.",
        "url": "https://cv.cs.nthu.edu.tw/php/callforpaper/category.php?net_id=14",
        "type": "video_sequences",
        "classes": ["normal", "talking", "yawning", "nodding", "sleepy"],
    },
    "yawdd": {
        "name": "YawDD (Yawning Detection Dataset)",
        "description": "322 videos of male/female drivers in vehicles with normal talking, yawning, and driving behaviors.",
        "url": "https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
        "type": "video_sequences",
        "classes": ["normal", "yawning", "talking"],
    }
}


def print_catalog():
    """Display information about available benchmark datasets."""
    print("=" * 70)
    print("🎯 REPUTABLE DRIVER DROWSINESS BENCHMARK DATASETS")
    print("=" * 70)
    for key, info in DATASET_CATALOG.items():
        print(f"\n[{key.upper()}] - {info['name']}")
        print(f"  • Description: {info['description']}")
        print(f"  • Data Type:   {info['type']}")
        print(f"  • Classes:     {', '.join(info['classes'])}")
        print(f"  • Official URL:{info['url']}")
    print("\n" + "=" * 70)


def create_sample_dataset(target_dir: str = "data/raw/sample"):
    """
    Creates a synthetic/placeholder dataset structure with verified samples
    to allow immediate end-to-end testing of the pipeline without waiting for
    large multi-gigabyte downloads.
    """
    import cv2
    import numpy as np

    os.makedirs(os.path.join(target_dir, "awake"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "drowsy"), exist_ok=True)

    print(f"Generating synthetic baseline sample frames in '{target_dir}' for pipeline verification...")

    # We generate test placeholder images (640x480) with simulated face shapes
    for state, folder in [("awake", "awake"), ("drowsy", "drowsy")]:
        for i in range(20):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Background
            img[:] = (40, 40, 40)
            # Oval face
            cv2.ellipse(img, (320, 240), (120, 160), 0, 0, 360, (180, 200, 220), -1)
            # Eyes
            if state == "awake":
                # Open eyes
                cv2.ellipse(img, (270, 200), (20, 12), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (270, 200), 6, (50, 50, 50), -1)
                cv2.ellipse(img, (370, 200), (20, 12), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (370, 200), 6, (50, 50, 50), -1)
                # Mouth closed
                cv2.line(img, (290, 310), (350, 310), (50, 50, 150), 3)
            else:
                # Closed eyes (horizontal line)
                cv2.line(img, (250, 200), (290, 200), (50, 50, 50), 3)
                cv2.line(img, (350, 200), (390, 200), (50, 50, 50), 3)
                # Yawning mouth (big open circle)
                cv2.ellipse(img, (320, 310), (25, 35), 0, 0, 360, (20, 20, 100), -1)

            filename = os.path.join(target_dir, folder, f"sample_{i:03d}.jpg")
            cv2.imwrite(filename, img)

    print(f"Sample dataset ready at: {target_dir}")


if __name__ == "__main__":
    print_catalog()
    create_sample_dataset()
