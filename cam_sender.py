#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import zmq

PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind("tcp://127.0.0.1:5555")   # aynı cihaz içi IPC
    time.sleep(0.3)  # SUB bağlansın diye küçük bekleme

    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[SENDER][HATA] Kamera açılamadı.")
        return

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # hız/kalite dengesi
    print("[SENDER] Kamera başladı. Yayın: tcp://127.0.0.1:5555 (CTRL+C ile çık)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            # topic + payload
            sock.send_multipart([b"frame", jpg.tobytes()])
    except KeyboardInterrupt:
        print("\n[SENDER] Çıkılıyor...")
    finally:
        cap.release()
        sock.close()
        ctx.term()
        print("[SENDER] Bitti.")

if __name__ == "__main__":
    main()
