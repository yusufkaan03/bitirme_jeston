#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def main():
    print("[INFO] GStreamer ile canlı görüntü başlatılıyor...")
    print("[INFO] Pencereyi kapatmak veya CTRL+C ile çıkmak yeterli.")

    # ÇALIŞAN KOMUT (sadece görüntü):
    # gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! nvvidconv ! videoconvert ! ximagesink

    pipeline_cmd = (
        "gst-launch-1.0 "
        "nvarguscamerasrc ! "
        "'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! "
        "nvvidconv ! "
        "videoconvert ! "
        "ximagesink"
    )

    try:
        proc = subprocess.Popen(pipeline_cmd, shell=True)
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
