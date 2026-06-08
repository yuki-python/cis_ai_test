# MIT License
import cv2
import time
import os

def run_camera(output_dir="camera_frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Camera not found")
        return

    print("Camera started. Press Ctrl+C to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ts = time.time()
        filename = f"{output_dir}/{ts:.6f}.jpg"
        cv2.imwrite(filename, frame)

        time.sleep(0.033)  # 約30FPS

if __name__ == "__main__":
    run_camera()
