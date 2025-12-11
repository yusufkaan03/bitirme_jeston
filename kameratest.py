#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import datetime
from pathlib import Path

def main():
    output_dir = Path.home() / "bitirme_jeston" / "kayitlar"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"kamera_{ts}.mp4"

    print(f"[INFO] Kayıt klasörü : {output_dir}")
    print(f"[INFO] Kayıt dosyası : {output_file}")
    print("[INFO] Kayıt başlatılıyor...")
    print("[INFO] Çıkmak için CTRL + C")

    # NVIDIA Jetson resmi örneklere çok benzeyen pipeline:
    # nvarguscamerasrc -> 1280x720@30 NV12 -> H264 encode -> MP4 dosya
    pipeline_cmd = (
        "gst-launch-1.0 -e "
        "nvarguscamerasrc ! "
        "'video/x-raw(memory:NVMM), "
        "width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)30/1' ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "qtmux ! "
        f"filesink location={output_file}"
    )

    try:
        # shell=True: komutu terminalde yazmışsın gibi çalıştırıyoruz
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
