#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
from pathlib import Path

def main():
    # Kayıtları kaydedeceğimiz klasör (istersen root'a da koyabiliriz)
    output_dir = Path.home() / "bitirme_jeston" / "kayitlar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dosya adı: kamera_YYYYmmdd_HHMMSS.mp4
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"kamera_kod_{ts}.mp4"

    print(f"[INFO] Kayıt klasörü : {output_dir}")
    print(f"[INFO] Kayıt dosyası : {output_file}")
    print("[INFO] Kayıt başlatılıyor (Python içinden)...")
    print("[INFO] Durdurmak için CTRL + C")

    # Terminalde çalışan KOMUTU birebir string olarak yazıyoruz
    # Dikkat: video/x-raw(...) kısmındaki tek tırnaklar aynen korunuyor
    pipeline_cmd = (
        "gst-launch-1.0 -e "
        "nvarguscamerasrc ! "
        "'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "qtmux ! "
        f"filesink location={output_file}"
    )

    try:
        # shell=True => sanki terminale bu komutu aynen yazmışsın gibi
        proc = subprocess.Popen(pipeline_cmd, shell=True)
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
