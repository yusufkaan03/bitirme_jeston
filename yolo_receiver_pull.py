#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import zmq
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect("tcp://127.0.0.1:5556")

    model = YOLO("yolov8n-seg.pt")
    print("[RECV] PULL başladı. Çıkış: q")

    fps = 0.0
    last_t = time.time()
    frame_id = 0

    try:
        while True:
            # En günceli almak için queue drain
            msg = None
            while True:
                try:
                    msg = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            if msg is None:
                msg = sock.recv_multipart()

            header, payload = msg
            w_str, h_str, ts_str = header.decode().split(",")
            w, h = int(w_str), int(h_str)
            ts = float(ts_str)

            frame = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3))

            # daha da hafif: imgsz=256
            results = model.predict(frame, imgsz=256, conf=0.35, verbose=False)
            r = results[0]

            vis = frame.copy()
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            now = time.time()
            dt = now - last_t
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else (0.9 * fps + 0.1 * inst)
            last_t = now
            frame_id += 1

            lat_ms = (now - ts) * 1000.0

            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"LAT: {lat_ms:.0f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            vis_big = cv2.resize(vis, (1280, 720))
            cv2.imshow("YOLOv8 (PULL, low-lat)", vis_big)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        print("[RECV] Bitti.")

if __name__ == "__main__":
    main()
