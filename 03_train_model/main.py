# CSIPoseModelを変えただけ。

# lib/train_model.py
# CSI(2デバイス) + Labelme骨格 で 3D CNN + Temporal Transformer + GCN
# 5-fold クロスバリデーション + test固定 + lossグラフ保存 + testデータコピー
# person_exist のみ学習、person_no_exist は背景差分用

import os
import glob
import json
import shutil
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold

# =========================
# 設定（main.py から dataset_root を受け取る）
# =========================

NUM_SUBCARRIERS = 128
TIME_WINDOW = 1
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-3
MAX_DIFF = 1.0

# Mediapipe の 33 keypoints のうち使用する ID
KP_IDS = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 20, 22]
NUM_KP = len(KP_IDS)

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# ユーティリティ関数（その1）
# =========================
def parse_csi_line(line: str) -> np.ndarray:
    """
    ESP32 CSI Tool 形式の 1 行をパースして、
    buf= の後ろの 128 個のサブキャリア値だけを抽出する。
    """
    line = line.strip()

    # buf= の位置を探す
    if "buf=" not in line:
        return np.array([], dtype=np.float32)

    # buf= の後ろを取り出す
    buf_part = line.split("buf=")[1]

    # カンマ区切りで数値化
    parts = buf_part.split(",")
    values = []
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        try:
            values.append(float(p))
        except ValueError:
            continue

    return np.array(values, dtype=np.float32)

def load_csi_from_crv(path: str) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        arr = parse_csi_line(line)
        if arr.size > 0:
            return arr.astype(np.float32)

    # もし何も取れなかったらゼロベクトル
    return np.zeros((128,), dtype=np.float32)


def load_pose_from_labelme(json_path: str) -> np.ndarray:
    """Labelme JSON から keypoint を読み込む（正規化済み）"""
    if not os.path.exists(json_path):
        return np.zeros((2 * NUM_KP,), dtype=np.float32)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W = data.get("imageWidth", 640)
    H = data.get("imageHeight", 480)

    kp_dict = {}
    for s in data["shapes"]:
        label = s.get("label", "")
        if not label.startswith("kp_"):
            continue
        try:
            idx = int(label.split("_")[1])
        except ValueError:
            continue

        x, y = s["points"][0]
        kp_dict[idx] = (x / W, y / H)

    pose = []
    for k in KP_IDS:
        if k in kp_dict:
            pose.extend(kp_dict[k])
        else:
            pose.extend([0.0, 0.0])

    return np.array(pose, dtype=np.float32)


# =========================
# ユーティリティ関数（その2）
# =========================

def smooth_subcarriers(csi_vec: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """CSI のサブキャリアを移動平均で平滑化"""
    if kernel_size <= 1:
        return csi_vec

    pad = kernel_size // 2
    padded = np.pad(csi_vec, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def parse_ts_from_path(path: str) -> float:
    """ファイル名（1234567890.jpg）から timestamp を抽出"""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return float(name)



class CSIPoseDataset(Dataset):
    """
    person_exist のみを教師データに使い、
    person_no_exist の CSI から背景平均を作って差分を取る。
    """

    def __init__(self, dataset_root: str):
        super().__init__()

        self.samples: List[Tuple[str, str, str]] = []

        # === 背景CSI（no_exist）を読み込んで平均を取る ===
        no_dev0_dir = os.path.join(dataset_root, "person_no_exist", "csi_frames", "device_0")
        no_dev1_dir = os.path.join(dataset_root, "person_no_exist", "csi_frames", "device_1")

        no0_files = sorted(glob.glob(os.path.join(no_dev0_dir, "*.crv")))
        no1_files = sorted(glob.glob(os.path.join(no_dev1_dir, "*.crv")))

        bg0_list = [load_csi_from_crv(f) for f in no0_files]
        bg1_list = [load_csi_from_crv(f) for f in no1_files]

        self.bg0 = np.mean(bg0_list, axis=0) if bg0_list else np.zeros((NUM_SUBCARRIERS,), dtype=np.float32)
        self.bg1 = np.mean(bg1_list, axis=0) if bg1_list else np.zeros((NUM_SUBCARRIERS,), dtype=np.float32)

        print(f"[CSIPoseDataset] background CSI: dev0={len(bg0_list)}, dev1={len(bg1_list)}")


        # === person_exist の対応付け ===
        exist_cam_dir = os.path.join(dataset_root, "person_exist", "camera_frames")
        exist_dev0_dir = os.path.join(dataset_root, "person_exist", "csi_frames", "device_0")
        exist_dev1_dir = os.path.join(dataset_root, "person_exist", "csi_frames", "device_1")

        exist_jpgs = sorted(glob.glob(os.path.join(exist_cam_dir, "*.jpg")))
        dev0_files = sorted(glob.glob(os.path.join(exist_dev0_dir, "*.crv")))
        dev1_files = sorted(glob.glob(os.path.join(exist_dev1_dir, "*.crv")))

        dev0_ts = [parse_ts_from_path(f) for f in dev0_files]
        dev1_ts = [parse_ts_from_path(f) for f in dev1_files]

        for img_path in exist_jpgs:
            t_img = parse_ts_from_path(img_path)
            json_path = os.path.splitext(img_path)[0] + ".json"

            if not dev0_ts or not dev1_ts:
                continue

            # dev0 の最も近い timestamp
            diffs0 = [abs(t_img - t) for t in dev0_ts]
            idx0 = min(range(len(diffs0)), key=lambda i: diffs0[i])
            if diffs0[idx0] > MAX_DIFF:
                continue

            # dev1 の最も近い timestamp
            diffs1 = [abs(t_img - t) for t in dev1_ts]
            idx1 = min(range(len(diffs1)), key=lambda i: diffs1[i])
            if diffs1[idx1] > MAX_DIFF:
                continue

            dev0_crv = dev0_files[idx0]
            dev1_crv = dev1_files[idx1]

            self.samples.append((dev0_crv, dev1_crv, json_path))

        print(f"[CSIPoseDataset] person_exist samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dev0_crv, dev1_crv, json_path = self.samples[idx]

        # === CSI 読み込み ===
        csi0 = load_csi_from_crv(dev0_crv)
        csi1 = load_csi_from_crv(dev1_crv)

        # === 背景差分 ===
        csi0 = csi0 - self.bg0
        csi1 = csi1 - self.bg1

        # === 平滑化 ===
        csi0 = smooth_subcarriers(csi0)
        csi1 = smooth_subcarriers(csi1)

        # shape: (2, 1, 128, 1)
        csi = np.stack([csi0, csi1], axis=0)
        csi = csi[:, np.newaxis, :, np.newaxis]

        # === Pose 読み込み ===
        pose = load_pose_from_labelme(json_path)

        return (
            torch.from_numpy(csi).float(),
            torch.from_numpy(pose).float(),
            (dev0_crv, dev1_crv, json_path)
        )




# =========================
# モデル本体（軽量1D-CNN + MLP）
# =========================
class CSIPoseModel(nn.Module):
    """CSI → Pose（軽量1D-CNN版）"""
    def __init__(self, num_kp=NUM_KP):
        super().__init__()

        # Conv層を2層に削減（32→32）
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # MLP部（Dropout強め）
        self.fc = nn.Sequential(
            nn.Linear(32 * 128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_kp * 2)
        )

    def forward(self, x):
        # x: (B, 2, 1, 128, 1)
        B = x.shape[0]
        x = x.squeeze(2).squeeze(3)  # (B, 2, 128)
        x = self.conv(x)             # (B, 32, 128)
        x = x.view(B, -1)            # (B, 32*128)
        out = self.fc(x)
        return out




def pose_loss_with_bones(pred, gt, num_kp=NUM_KP, lambda_bone=0.1):
    """
    pred, gt: (B, 2*num_kp) の正規化座標 (0〜1)
    """
    B = pred.shape[0]

    # 位置の L2 + L1
    l2 = torch.mean((pred - gt) ** 2)
    l1 = torch.mean(torch.abs(pred - gt))
    pos_loss = 0.5 * l2 + 0.5 * l1

    # 骨長 loss
    pred_xy = pred.view(B, num_kp, 2)
    gt_xy   = gt.view(B, num_kp, 2)

    bone_losses = []
    for i, j in BONES:
        pred_len = torch.norm(pred_xy[:, i] - pred_xy[:, j], dim=1)  # (B,)
        gt_len   = torch.norm(gt_xy[:, i]   - gt_xy[:, j],   dim=1)
        bone_losses.append((pred_len - gt_len) ** 2)

    if bone_losses:
        bone_loss = torch.mean(torch.stack(bone_losses, dim=0))
    else:
        bone_loss = 0.0 * pos_loss

    return pos_loss + lambda_bone * bone_loss



# =========================
# 学習・評価関連
# =========================

def save_loss_curve(train_losses, val_losses, fold, out_dir):
    """train/val loss の推移を PNG で保存"""
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    out_path = os.path.join(out_dir, f"loss_curve_fold{fold}.png")
    plt.savefig(out_path)
    plt.close()

    print("Saved loss curve:", out_path)


def copy_test_files(test_samples, fold, out_dir):
    """test データを model_evaluate/fold_X にコピー"""
    out_dir = os.path.join(out_dir, "model_evaluation", f"fold_{fold}")
    cam_dir = os.path.join(out_dir, "camera_frames")
    dev0_dir = os.path.join(out_dir, "csi_frames", "device_0")
    dev1_dir = os.path.join(out_dir, "csi_frames", "device_1")

    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(dev0_dir, exist_ok=True)
    os.makedirs(dev1_dir, exist_ok=True)

    for dev0, dev1, json_path in test_samples:
        img_path = json_path.replace(".json", ".jpg")

        if os.path.exists(img_path):
            shutil.copy(img_path, cam_dir)
        if os.path.exists(json_path):
            shutil.copy(json_path, cam_dir)

        shutil.copy(dev0, dev0_dir)
        shutil.copy(dev1, dev1_dir)

    print(f"[fold {fold}] Copied test files to:", out_dir)


def train_one_fold(model, train_loader, val_loader, fold, out_dir):
    """1 fold 分の学習"""
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ===== train =====
        model.train()
        total_loss = 0
        count = 0

        for csi, pose, _ in train_loader:
            csi = csi.to(DEVICE)
            pose = pose.to(DEVICE)

            optimizer.zero_grad()
            pred = model(csi)
            # loss = criterion(pred, pose)
            loss = pose_loss_with_bones(pred, pose) 
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * csi.size(0)
            count += csi.size(0)

        train_loss = total_loss / count
        train_losses.append(train_loss)

        # ===== validation =====
        model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for csi, pose, _ in val_loader:
                csi = csi.to(DEVICE)
                pose = pose.to(DEVICE)
                pred = model(csi)
                # loss = criterion(pred, pose)
                loss = pose_loss_with_bones(pred, pose) 
                total_loss += loss.item() * csi.size(0)
                count += csi.size(0)

        val_loss = total_loss / count
        val_losses.append(val_loss)

        print(f"[Fold {fold}] Epoch {epoch}/{NUM_EPOCHS} - train: {train_loss:.6f}, val: {val_loss:.6f}")

    # loss グラフ保存
    save_loss_curve(train_losses, val_losses, fold, out_dir)

    # モデル保存
    ckpt_path = os.path.join(out_dir, f"csi_pose_fold{fold}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)


def evaluate_test(model, test_loader):
    """test データで評価"""
    # criterion = nn.MSELoss()
    model.eval()

    total_loss = 0
    count = 0

    with torch.no_grad():
        for csi, pose, _ in test_loader:
            csi = csi.to(DEVICE)
            pose = pose.to(DEVICE)
            pred = model(csi)
            loss = pose_loss_with_bones(pred, pose)

            total_loss += loss.item() * csi.size(0)
            count += csi.size(0)

    return total_loss / count




# =========================
# メイン処理（7:2:1 分割 + 5-fold）
# =========================

def main(dataset_root):
    print(f"[train_model] dataset_root = {dataset_root}")

    # ===== データセット読み込み =====
    dataset = CSIPoseDataset(dataset_root)
    N = len(dataset)

    if N == 0:
        print("Dataset is empty.")
        return

    # ===== 7:2:1 のうち test = 10% を最初に固定 =====
    indices = list(range(N))
    random.shuffle(indices)

    test_size = int(N * 0.1)
    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]

    print(f"Total: {N}, test: {len(test_indices)}, train_val: {len(train_val_indices)}")

    # ===== test データは固定 =====
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # test データのファイルパス一覧（コピー用）
    test_samples = [dataset.samples[i] for i in test_indices]

    # ===== train_val（90%）を 5-fold に分割 =====
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"========== Fold {fold} ==========")

        train_ids = [train_val_indices[i] for i in train_idx]
        val_ids   = [train_val_indices[i] for i in val_idx]

        train_subset = Subset(dataset, train_ids)
        val_subset   = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # ===== モデル作成 =====
        model = CSIPoseModel(num_kp=NUM_KP).to(DEVICE)

        # ===== 学習 =====
        out_dir = dataset_root
        train_one_fold(model, train_loader, val_loader, fold, out_dir)

        # ===== テスト評価 =====
        test_loss = evaluate_test(model, test_loader)
        print(f"[Fold {fold}] Test Loss = {test_loss:.6f}")

        # ===== test データを fold_X にコピー =====
        copy_test_files(test_samples, fold, out_dir)

    print("=== All folds completed ===")



if __name__ == "__main__":
    main("./dataset")