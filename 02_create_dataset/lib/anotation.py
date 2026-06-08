# annotation.py
# Mediapipe PoseLandmarker を使って自動で Labelme 形式の骨格ラベルを生成

import os
import csv
import json
import cv2
import numpy as np
import mediapipe as mp

PAIRS_CSV = "pairs_multi_device.csv"
OUT_DIR = "labelme_json"

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"

def load_pairs():
    rows = []
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_labelme(image_path, keypoints):
    h, w = keypoints["image_shape"]
    shapes = []
    for i, (x, y, v) in enumerate(keypoints["points"]):
        if v == 0:
            continue
        shapes.append({
            "label": f"kp_{i}",
            "points": [[float(x), float(y)]],
            "group_id": None,
            "shape_type": "point",
            "flags": {}
        })
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    return data

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_pairs()

    BaseOptions = python.BaseOptions
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=False
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    for row in pairs:
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            continue

        pts = []
        for lm in result.pose_landmarks[0]:
            x = lm.x * w
            y = lm.y * h
            v = 1 if lm.visibility > 0.5 else 0
            pts.append((x, y, v))

        kp = {"image_shape": (h, w), "points": pts}
        labelme_json = to_labelme(img_path, kp)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUT_DIR, base + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelme_json, f, ensure_ascii=False, indent=2)

        print("saved labelme:", out_path)
