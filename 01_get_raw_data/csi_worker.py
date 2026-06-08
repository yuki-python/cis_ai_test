# MIT License
import serial
import serial.tools.list_ports
import time
import os

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if ("USB" in p.description or 
            "UART" in p.description or 
            "Silicon" in p.description or 
            "CH340" in p.description):
            return p.device
    return None

def run_csi(baud=921600, output_dir="csi_frames"):
    os.makedirs(output_dir, exist_ok=True)

    port = find_esp32_port()
    if port is None:
        print("ESP32 の COM ポートが見つかりません")
        return

    print(f"CSI logging started on {port}. Press Ctrl+C to stop.")

    ser = serial.Serial(port, baud, timeout=0.01)

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            ts = time.time()
            filename = f"{output_dir}/{ts:.6f}.crv"
            with open(filename, "w") as f:
                f.write(line)
            print("CSI frame saved:", ts)

if __name__ == "__main__":
    run_csi()
