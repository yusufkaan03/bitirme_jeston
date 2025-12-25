#!/usr/bin/env python3
# -*- coding: utf-8 -*-
















import time
import zmq
import numpy as np
import cv2
from ultralytics import YOLO
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

PIPELINE = (
    "nvarguscamerasrc sensor-mode=2 ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:5556")  # yeni port
    time.sleep(0.2)

    pipeline = Gst.parse_launch(PIPELINE)
    appsink = pipeline.get_by_name("sink")
    if appsink is None:
        print("[SENDER][HATA] appsink yok.")
        return

    pipeline.set_state(Gst.State.PLAYING)
    print("[SENDER] PUSH yayın başladı: tcp://127.0.0.1:5556 (CTRL+C)")

    # FPS cap
    target_fps = 15.0
    min_dt = 1.0 / target_fps
    last_send = 0.0

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 2_000_000_000)
            if sample is None:
                continue

            now = time.time()
            if now - last_send < min_dt:
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            w = int(s.get_value("width"))
            h = int(s.get_value("height"))

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue

            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            buf.unmap(mapinfo)

            ts = time.time()
            header = f"{w},{h},{ts}".encode()

            # tek multipart mesaj
            sock.send_multipart([header, frame.tobytes()])
            last_send = now

    except KeyboardInterrupt:
        print("\n[SENDER] Çıkılıyor...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        sock.close()
        ctx.term()
        print("[SENDER] Bitti.")





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
