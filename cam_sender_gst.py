#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import zmq
import numpy as np
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

if __name__ == "__main__":
    main()
