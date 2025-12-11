#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time

# Jetson CSI kamera pipeline
pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink"
)

print("[INFO] Kamera açılıyor...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[HATA] Kamera açılamadı.")
    exit()

print("[INFO] Kamera hazır. Çıkmak için 'q' ya da ESC.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Uyarı] Frame okunamadı.")
        break

    # ---- ŞU AN BURADA YOLO ÇALIŞACAK ----
    # örnek olarak FPS yazdıralım:
    cv2.putText(frame, "YOLO test hazir", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Test - Jetson CSI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
