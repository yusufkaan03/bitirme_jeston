#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
from pathlib import Path

def main():
    # Kayıt dosyasını kaydedeceğimiz yer
    output_dir = Path.home() / "bitirme_jeston" / "kayitlar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dosya adı: kamera_YYYYmmdd_HHMMSS.mp4
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"kamera_{ts}.mp4"

    print(f"[INFO] Kayıt klasörü : {output_dir}")
    print(f"[INFO] Kayıt dosyası : {output_file}")

    # GStreamer pipeline:
    # CSI kamera (nvarguscamerasrc) -> H264 encode -> MP4 dosya
    #
    # Çözünürlük: 1280x720
    # FPS       : 30
    #
    # Kaydı durdurmak için: CTRL + C
    pipeline = [
        "gst-launch-1.0",
        "nvarguscamerasrc",
        "!",
        "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1",
        "!",
        "nvv4l2h264enc",          # Gerekirse omxh264enc ile de deneyebiliriz
        "bitrate=8000000",        # 8 Mbps
        "!",
        "h264parse",
        "!",
        "qtmux",
        "!",
        f"filesink location={str(output_file)}"
    ]

    print("[INFO] Kayıt başlatılıyor...")
    print("[INFO] Çıkmak için CTRL + C")

    try:
        # -e parametresine burada gerek yok; Python CTRL+C ile sonlandıracak
        proc = subprocess.Popen(pipeline)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Kayıt durduruluyor (CTRL + C)…")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        print("[INFO] Bitti.")

if __name__ == "__main__":
    main()
