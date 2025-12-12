#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time

# Conda içinde gi bulunmazsa (Ubuntu'da çoğunlukla burada olur)
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np
import cv2  # sadece ekrana basmak için (kamera açmıyoruz)
from ultralytics import YOLO

Gst.init(None)

PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
)

def main():
    print("[INFO] GStreamer appsink + YOLOv8-seg live başlıyor. Çıkış: q")
    model = YOLO("yolov8n-seg.pt")  # istersen yolov8s-seg.pt

    pipeline = Gst.parse_launch(PIPELINE)
    appsink = pipeline.get_by_name("sink")
    if appsink is None:
        print("[HATA] appsink bulunamadı.")
        return

    pipeline.set_state(Gst.State.PLAYING)

    last_t = time.time()
    fps = 0.0

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 1_000_000)  # 1s timeout (ns)
            if sample is None:
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            w = s.get_value("width")
            h = s.get_value("height")

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue

            # BGR packed: h*w*3
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            buf.unmap(mapinfo)

            # YOLOv8-seg inference
            results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
            vis = results[0].plot()

            # FPS hesapla
            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            last_t = now

            cv2.putText(vis, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("YOLOv8-Seg (CSI via GStreamer appsink)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        print("[INFO] Bitti.")

if __name__ == "__main__":
    main()
