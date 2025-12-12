#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import zmq
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

# Daha stabil olsun diye 30 fps'e kilitlemek istersen sensor-mode=2 kullanabilirsin:
# 1920x1080@30 -> mode 2 (senin listende öyle görünüyor)
# Aşağıdaki pipeline 1080p30 alır, 720p'ye indirir.
PIPELINE = (
    "nvarguscamerasrc sensor-mode=2 ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)   # kuyruk şişmesin
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:5555")
    time.sleep(0.3)

    pipeline = Gst.parse_launch(PIPELINE)
    appsink = pipeline.get_by_name("sink")
    if appsink is None:
        print("[SENDER][HATA] appsink bulunamadı.")
        return

    pipeline.set_state(Gst.State.PLAYING)
    print("[SENDER] Yayın başladı: tcp://127.0.0.1:5555 (CTRL+C ile çık)")

    count = 0
    t0 = time.time()

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 2_000_000_000)  # 2s timeout
            if sample is None:
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

            header = f"{w},{h}".encode()
            sock.send_multipart([b"raw", header, frame.tobytes()])

            count += 1
            if count % 60 == 0:
                dt = time.time() - t0
                print(f"[SENDER] {count} frame gönderildi ({count/dt:.1f} fps)")

    except KeyboardInterrupt:
        print("\n[SENDER] Çıkılıyor...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        sock.close()
        ctx.term()
        print("[SENDER] Bitti.")

if __name__ == "__main__":
    main()
