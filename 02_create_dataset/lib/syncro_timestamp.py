# sync_multi_device.py
# prefix 付きファイル名 + 複数デバイスに対応した同期スクリプト
# 入力は merge_datasets.py が作った merged_dataset

import os
import glob
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "merged_dataset")  
OUT_CSV = "pairs_multi_device.csv"
MAX_DIFF = 1.0  # 秒

def parse_ts(path):
    """
    prefix_22.313149.jpg → 22.313149 を抽出する
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    parts = name.split("_")
    ts_str = parts[-1]

    try:
        return float(ts_str)
    except:
        return None

def sync_one_session(cam_dir, csi_base_dir, person_flag):
    img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
    img_files = [os.path.abspath(f) for f in img_files]

    device_dirs = sorted(glob.glob(os.path.join(csi_base_dir, "device_*")))
    num_devices = len(device_dirs)

    dev_files = []
    dev_ts = []

    for d in device_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.crv")))
        files = [os.path.abspath(f) for f in files]
        ts = [parse_ts(f) for f in files]
        dev_files.append(files)
        dev_ts.append(ts)

    pairs = []

    for img in img_files:
        t_img = parse_ts(img)
        if t_img is None:
            continue

        matched_paths = []
        matched_ts = []

        for dev_idx in range(num_devices):
            ts_list = dev_ts[dev_idx]
            files_list = dev_files[dev_idx]

            if not ts_list:
                break

            diffs = [abs(t_img - t) for t in ts_list]
            idx = min(range(len(diffs)), key=lambda i: diffs[i])

            if diffs[idx] > MAX_DIFF:
                break

            matched_paths.append(files_list[idx])
            matched_ts.append(ts_list[idx])

        if len(matched_paths) == num_devices:
            pairs.append(
                [img] + matched_paths + [t_img] + matched_ts + [person_flag]
            )

    return pairs

def main():
    all_pairs = []

    exist_cam = os.path.join(ROOT_DIR, "person_exist", "camera_frames")
    exist_csi = os.path.join(ROOT_DIR, "person_exist", "csi_frames")

    no_cam = os.path.join(ROOT_DIR, "person_no_exist", "camera_frames")
    no_csi = os.path.join(ROOT_DIR, "person_no_exist", "csi_frames")

    print("[person_exist] syncing...")
    exist_pairs = sync_one_session(exist_cam, exist_csi, person_flag=1)
    all_pairs.extend(exist_pairs)

    print("[person_no_exist] syncing...")
    no_pairs = sync_one_session(no_cam, no_csi, person_flag=0)
    all_pairs.extend(no_pairs)

    print("Total matched:", len(all_pairs))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        if not all_pairs:
            print("No pairs found. CSV will be empty.")
            return

        num_devices = (len(all_pairs[0]) - 2 - 1) // 2
        header = ["image_path"]
        header += [f"dev{i}_csi_path" for i in range(num_devices)]
        header += ["t_image"]
        header += [f"t_dev{i}" for i in range(num_devices)]
        header += ["person_exist"]

        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_pairs)

    print("Saved:", OUT_CSV)
