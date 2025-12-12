#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import zmq
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)   # sadece en son frame
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.connect("tcp://127.0.0.1:5555")
    sock.setsockopt_string(zmq.SUBSCRIBE, "raw")

    model = YOLO("yolov8n-seg.pt")
    print("[RECV] YOLOv8-seg başladı. Çıkış: q")

    fps = 0.0
    last_t = time.time()
    i = 0

    try:
        while True:
            topic, header, payload = sock.recv_multipart()
            w_str, h_str = header.decode().split(",")
            w, h = int(w_str), int(h_str)

            frame = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3))

            # hafiflet
            frame_small = cv2.resize(frame, (640, 360))

            results = model.predict(frame_small, imgsz=416, conf=0.25, verbose=False)
            vis = results[0].plot()
            vis = cv2.resize(vis, (w, h))

            now = time.time()
            dt = now - last_t
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else (0.9 * fps + 0.1 * inst)
            last_t = now
            i += 1

            cv2.putText(vis, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(vis, f"frame: {i}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("YOLOv8-Seg (RAW IPC)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        print("[RECV] Bitti.")

if __name__ == "__main__":
    main()
