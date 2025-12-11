#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def main():
    print("[INFO] GStreamer ile CANLI görüntü başlatılıyor (kayıt yok, OpenCV yok)...")
    print("[INFO] Pencereyi kapatmak veya terminalde CTRL+C yapmak yeterli.")

    # Terminalde çalışan komutun bire bir Python listesi hali:
    # gst-launch-1.0 nvarguscamerasrc ! \
    #   "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1" ! \
    #   nvvidconv ! videoconvert ! ximagesink

    cmd = [
        "gst-launch-1.0",
        "nvarguscamerasrc",
        "!",
        "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1",
        "!",
        "nvvidconv",
        "!",
        "videoconvert",
        "!",
        "ximagesink",
    ]

    try:
        # shell=False → '(' karakteri sh'e gitmiyor, direkt gst-launch'a gidiyor
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C ile çıkılıyor...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        print("[INFO] Bitti.")

if __name__ == "__main__":
    main()
