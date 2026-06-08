# main.py
# 前処理 → 同期 → アノテーション → データエクスポート → 学習
# すべてを自動で実行するパイプライン

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE_DIR, "lib")

# lib を import パスに追加
sys.path.append(LIB_DIR)

# 各処理を import
import merge_raw_data
import syncro_timestamp
import anotation
import export_synced_dataset

def main():
    print("\n=== Step 1: Merge raw datasets ===")
    merge_raw_data.merge()

    print("\n=== Step 2: Sync multi-device timestamps ===")
    syncro_timestamp.main()

    print("\n=== Step 3: Auto annotation (Mediapipe) ===")
    anotation.main()

    print("\n=== Step 4: Export synced dataset to model_learning ===")
    export_synced_dataset.main()

    print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
