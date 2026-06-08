# merge_datasets.py
# 複数セッションフォルダを1つの統合データセットにまとめる
# 出力先は「このスクリプトを実行したカレントディレクトリ配下の merged_dataset」

import os
import glob
import shutil

# 生データのルート（ここは raw_data 側）
SRC_ROOT = r"raw data directory"

# 出力先はカレントディレクトリ配下
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DST_ROOT = os.path.join(BASE_DIR, "merged_dataset")

SESSIONS = ["0", "1", "2", "3", "4"]
PERSON_NO_EXIST = "person_no_exist"

def ensure_dirs(base):
    os.makedirs(os.path.join(base, "person_exist", "camera_frames"), exist_ok=True)
    os.makedirs(os.path.join(base, "person_exist", "csi_frames", "device_0"), exist_ok=True)
    os.makedirs(os.path.join(base, "person_exist", "csi_frames", "device_1"), exist_ok=True)
    os.makedirs(os.path.join(base, "person_no_exist", "camera_frames"), exist_ok=True)
    os.makedirs(os.path.join(base, "person_no_exist", "csi_frames", "device_0"), exist_ok=True)
    os.makedirs(os.path.join(base, "person_no_exist", "csi_frames", "device_1"), exist_ok=True)

def copy_files(src_dir, dst_dir, prefix):
    if not os.path.exists(src_dir):
        return
    for f in glob.glob(os.path.join(src_dir, "*")):
        base = os.path.basename(f)
        dst = os.path.join(dst_dir, f"{prefix}_{base}")
        shutil.copy2(f, dst)

def merge():
    ensure_dirs(DST_ROOT)

    # person_exist セッションを統合
    for sess in SESSIONS:
        cam_dir = os.path.join(SRC_ROOT, sess, "camera_frames")
        dev0_dir = os.path.join(SRC_ROOT, sess, "csi_frames", "device_0")
        dev1_dir = os.path.join(SRC_ROOT, sess, "csi_frames", "device_1")

        if not os.path.exists(cam_dir):
            continue

        print(f"Merging session {sess}...")

        copy_files(cam_dir, os.path.join(DST_ROOT, "person_exist", "camera_frames"), sess)
        copy_files(dev0_dir, os.path.join(DST_ROOT, "person_exist", "csi_frames", "device_0"), sess)
        copy_files(dev1_dir, os.path.join(DST_ROOT, "person_exist", "csi_frames", "device_1"), sess)

    # person_no_exist を統合
    no_cam = os.path.join(SRC_ROOT, PERSON_NO_EXIST, "camera_frames")
    no_dev0 = os.path.join(SRC_ROOT, PERSON_NO_EXIST, "csi_frames", "device_0")
    no_dev1 = os.path.join(SRC_ROOT, PERSON_NO_EXIST, "csi_frames", "device_1")

    print("Merging person_no_exist...")
    copy_files(no_cam, os.path.join(DST_ROOT, "person_no_exist", "camera_frames"), "no")
    copy_files(no_dev0, os.path.join(DST_ROOT, "person_no_exist", "csi_frames", "device_0"), "no")
    copy_files(no_dev1, os.path.join(DST_ROOT, "person_no_exist", "csi_frames", "device_1"), "no")

    print("✅ Merge complete! ->", DST_ROOT)
