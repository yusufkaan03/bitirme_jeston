#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def main():
    # Jetson CSI kamera için GStreamer pipeline
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
    )

    print("[INFO] Kamera açılıyor...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı (GStreamer/OpenCV).")
        return

    print("[INFO] Kamera açık. Çıkmak için pencere aktifken 'q' ya da ESC'e bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Uyarı] Frame okunamadı, döngüden çıkılıyor.")
            break

        cv2.imshow("Jetson CSI Kamera - Canlı", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC veya q
            print("[INFO] Çıkış tuşuna basıldı, kapanıyor...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Bitti.")

if __name__ == "__main__":
    main()
