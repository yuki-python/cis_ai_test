# MIT License
import threading
import time

from camera_worker import run_camera
from csi_worker import run_csi
from ping_worker import run_ping

def main():
    t1 = threading.Thread(target=run_camera, daemon=True)
    t2 = threading.Thread(target=run_csi, daemon=True)
    t3 = threading.Thread(target=run_ping, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    print("Camera + CSI + Ping started (threading). Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
