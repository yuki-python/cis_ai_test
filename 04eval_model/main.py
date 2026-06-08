import os
import json
import numpy as np
import torch
import cv2
import csv
import torch.nn as nn
import glob
from torch.utils.data import Dataset, DataLoader

DATASET_ROOT = "datasets"
OUT_DIR = "inference_outputs"
MODEL_PATH = "csi_pose_fold5.pth"
SYNCRO_TABLE = "../create_datasets/pairs_multi_device.csv"

# =========================
# モデル定義（model_learning と同じものを貼る）
# =========================
# Mediapipe の 33 keypoints のうち使用する ID
KP_IDS = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 20, 22]
NUM_KP = len(KP_IDS)


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


def pose_loss_with_bones(pred, gt, num_kp=14, lambda_bone=0.1):
    B = pred.shape[0]

    # L1 + L2
    l2 = torch.mean((pred - gt) ** 2)
    l1 = torch.mean(torch.abs(pred - gt))
    pos_loss = 0.5 * l2 + 0.5 * l1

    pred_xy = pred.view(B, num_kp, 2)
    gt_xy   = gt.view(B, num_kp, 2)

    bone_losses = []
    for i, j in BONES:
        pred_len = torch.norm(pred_xy[:, i] - pred_xy[:, j], dim=1)
        gt_len   = torch.norm(gt_xy[:, i] - gt_xy[:, j], dim=1)
        bone_losses.append((pred_len - gt_len) ** 2)

    bone_loss = torch.mean(torch.stack(bone_losses)) if bone_losses else 0.0
    return pos_loss + lambda_bone * bone_loss

# 学習時と同じ平滑化
def smooth_subcarriers(csi_vec: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return csi_vec
    pad = kernel_size // 2
    padded = np.pad(csi_vec, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KP = 14

# =========================
# CSI パーサー（学習時と同じ）
# =========================
def parse_csi_line(line: str) -> np.ndarray:
    if "buf=" not in line:
        return np.array([], dtype=np.float32)
    buf_part = line.split("buf=")[1]
    parts = buf_part.split(",")
    values = []
    for p in parts:
        p = p.strip()
        try:
            values.append(float(p))
        except:
            continue
    return np.array(values, dtype=np.float32)

def load_csi_from_crv(path: str) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        arr = parse_csi_line(line)
        if arr.size > 0:
            return arr.astype(np.float32)
    return np.zeros((128,), dtype=np.float32)

# =========================
# データセット
# =========================
class EvalDataset(Dataset):
    def __init__(self, root, pairs_csv, kp_ids):
        self.samples = []
        self.kp_ids = kp_ids

        # ===== 背景CSI（person_no_exist） =====
        no_dev0_dir = os.path.join(root, "person_no_exist", "csi_frames", "device_0")
        no_dev1_dir = os.path.join(root, "person_no_exist", "csi_frames", "device_1")

        no0_files = sorted(glob.glob(os.path.join(no_dev0_dir, "*.crv")))
        no1_files = sorted(glob.glob(os.path.join(no_dev1_dir, "*.crv")))

        bg0_list = [load_csi_from_crv(f) for f in no0_files]
        bg1_list = [load_csi_from_crv(f) for f in no1_files]

        self.bg0 = np.mean(bg0_list, axis=0) if bg0_list else np.zeros((128,), dtype=np.float32)
        self.bg1 = np.mean(bg1_list, axis=0) if bg1_list else np.zeros((128,), dtype=np.float32)

        print(f"[EvalDataset] background CSI: dev0={len(bg0_list)}, dev1={len(bg1_list)}")

        # ===== 評価対象（person_exist） =====
        cam_dir = os.path.join(root, "person_exist", "camera_frames")
        csi_dir = os.path.join(root, "person_exist", "csi_frames")

        with open(pairs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name  = os.path.basename(row["image_path"])
                dev0_name = os.path.basename(row["dev0_csi_path"])
                dev1_name = os.path.basename(row["dev1_csi_path"])

                local_img  = os.path.join(cam_dir, img_name)
                local_json = local_img.replace(".jpg", ".json")
                local_dev0 = os.path.join(csi_dir, "device_0", dev0_name)
                local_dev1 = os.path.join(csi_dir, "device_1", dev1_name)

                if (
                    os.path.exists(local_img)
                    and os.path.exists(local_json)
                    and os.path.exists(local_dev0)
                    and os.path.exists(local_dev1)
                ):
                    self.samples.append((local_img, local_json, local_dev0, local_dev1))

        print(f"[EvalDataset] Loaded {len(self.samples)} valid pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, dev0, dev1 = self.samples[idx]

        # ===== CSI 読み込み =====
        csi0 = load_csi_from_crv(dev0)
        csi1 = load_csi_from_crv(dev1)

        # ===== 背景差分（学習時と同じ） =====
        csi0 = csi0 - self.bg0
        csi1 = csi1 - self.bg1

        # ===== 平滑化（学習時と同じ） =====
        print("CSI mean:", np.mean(csi0), np.mean(csi1))    
        csi0 = smooth_subcarriers(csi0)
        csi1 = smooth_subcarriers(csi1)
        print("CSI mean:", np.mean(csi0), np.mean(csi1))    
        # shape: (2,1,128,1)
        csi = np.stack([csi0, csi1], axis=0)
        csi = csi[:, np.newaxis, :, np.newaxis]

        # ===== 正解 keypoints（正規化済み） =====
        kp = load_labelme_keypoints(json_path, self.kp_ids)  # (NUM_KP, 2), 0〜1

        return (
            torch.from_numpy(csi).float(),
            torch.from_numpy(kp).float(),  # tensor にしておく
            img_path
        )



# =========================
# 骨格描画
# =========================
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (8,11),(11,12),(12,13)
]


def load_labelme_keypoints(json_path, kp_ids):
    with open(json_path, "r") as f:
        ann = json.load(f)

    W = ann.get("imageWidth", 640)
    H = ann.get("imageHeight", 480)

    kp_dict = {}
    for shape in ann["shapes"]:
        label = shape["label"]
        if label.startswith("kp_"):
            idx = int(label.split("_")[1])
            x, y = shape["points"][0]
            kp_dict[idx] = (x / W, y / H)   # ★ 正規化を追加 ★

    keypoints = []
    for k in kp_ids:
        if k in kp_dict:
            keypoints.append(kp_dict[k])
        else:
            keypoints.append((0.0, 0.0))

    return np.array(keypoints, dtype=np.float32)




def draw_pose(img, kp, color=(0,255,0)):
    for x,y in kp:
        cv2.circle(img, (int(x),int(y)), 3, color, -1)
    for a,b in SKELETON:
        x1,y1 = kp[a]
        x2,y2 = kp[b]
        cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
    return img

# =========================
# 評価指標
# =========================
def compute_metrics(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=1)

    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)

    # PCK@0.05（画像サイズ 640x480 前提）
    threshold = 0.05 * 640
    pck = np.mean(dist < threshold)

    return mae, mse, pck





# =========================
# メイン処理
# =========================
def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    # モデル読み込み
    model = CSIPoseModel(num_kp=NUM_KP).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = EvalDataset(DATASET_ROOT, SYNCRO_TABLE, KP_IDS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (csi, gt_kp, img_paths) in enumerate(loader):
        print(f"Sample {i}: {img_paths[0]}")
        print("CSI mean:", torch.mean(csi).item(), "std:", torch.std(csi).item())

    all_mae, all_mse, all_pck = [], [], []

    for csi, gt_kp, img_paths in loader:
        csi = csi.to(DEVICE)
        with torch.no_grad():
            pred = model(csi)

        pred_np = pred.detach().cpu().numpy().reshape(-1, 2)
        gt_np   = gt_kp.detach().cpu().numpy().reshape(-1, 2)
        # print("pred_np:", pred_np)


        # img_paths はバッチサイズ1なので、0番目を取り出す
        img_path = img_paths[0]

        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        # 正規化 → ピクセル座標へ変換
        pred_px = pred_np * np.array([W, H])
        gt_px   = gt_np   * np.array([W, H])

        mae, mse, pck = compute_metrics(pred_px, gt_px)
        all_mae.append(mae)
        all_mse.append(mse)
        all_pck.append(pck)

        # ★ 推論骨格（元画像）
        pred_img = draw_pose(img.copy(), pred_px, (0,255,0))
        # ★ 正解骨格（黒背景）
        gt_img   = draw_pose(np.zeros_like(img), gt_px, (255,0,0))
        # ★ 推論骨格（黒背景）
        pred_black = draw_pose(np.zeros_like(img), pred_px, (0,255,0))
        # ★ 正解骨格（元画像）← 祐貴が追加で欲しいもの
        gt_on_img = draw_pose(img.copy(), gt_px, (255,0,0))

        # 出力フォルダ（画像ごと）
        base = os.path.basename(img_path).replace(".jpg","")
        out_dir = os.path.join(OUT_DIR, base)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{base}_pred.jpg"), pred_img)
        cv2.imwrite(os.path.join(out_dir, f"{base}_gt.jpg"), gt_img)
        cv2.imwrite(os.path.join(out_dir, f"{base}_pred_black.jpg"), pred_black)
        cv2.imwrite(os.path.join(out_dir, f"{base}_gt_on_img.jpg"), gt_on_img)



    print("=== Evaluation Results ===")
    print(f"MAE: {np.mean(all_mae):.4f}")
    print(f"MSE: {np.mean(all_mse):.4f}")
    print(f"PCK@0.05: {np.mean(all_pck):.4f}")

if __name__ == "__main__":
    main()
