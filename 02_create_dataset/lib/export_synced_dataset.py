# export_synced_dataset.py
# 時間同期 + アノテーション済みデータだけを model_learning/dataset に移動する

import os
import csv
import shutil

PAIRS_CSV = "pairs_multi_device.csv"
LABELME_DIR = "labelme_json"

OUT_ROOT = r"dataset directory"

def ensure_dirs():
    os.makedirs(os.path.join(OUT_ROOT, "person_exist", "camera_frames"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "person_exist", "csi_frames", "device_0"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "person_exist", "csi_frames", "device_1"), exist_ok=True)

def main():
    ensure_dirs()

    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row["person_exist"] != "1":
            continue

        img_path = row["image_path"]
        dev0_path = row["dev0_csi_path"]
        dev1_path = row["dev1_csi_path"]

        # Labelme JSON
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(LABELME_DIR, base + ".json")

        # コピー先
        cam_dst = os.path.join(OUT_ROOT, "person_exist", "camera_frames")
        dev0_dst = os.path.join(OUT_ROOT, "person_exist", "csi_frames", "device_0")
        dev1_dst = os.path.join(OUT_ROOT, "person_exist", "csi_frames", "device_1")

        # コピー
        if os.path.exists(img_path):
            shutil.copy2(img_path, cam_dst)

        if os.path.exists(json_path):
            shutil.copy2(json_path, cam_dst)

        if os.path.exists(dev0_path):
            shutil.copy2(dev0_path, dev0_dst)

        if os.path.exists(dev1_path):
            shutil.copy2(dev1_path, dev1_dst)

        print("Copied:", base)

    print("✅ Export complete! model_learning/dataset に同期済みデータを配置しました。")
