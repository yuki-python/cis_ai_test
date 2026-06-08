# MIT License
import subprocess
import time

def run_ping(target="192.168.2.100"):
    print(f"Pinging {target}. Press Ctrl+C to stop.")
    while True:
        subprocess.call(["ping", "-n", "1", target], stdout=subprocess.DEVNULL)
        time.sleep(0.01)

if __name__ == "__main__":
    run_ping()
