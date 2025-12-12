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
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind("tcp://127.0.0.1:5555")
    time.sleep(0.3)

    pipeline = Gst.parse_launch(PIPELINE)
    appsink = pipeline.get_by_name("sink")
    if appsink is None:
        print("[SENDER][HATA] appsink bulunamadı.")
        return

    pipeline.set_state(Gst.State.PLAYING)
    print("[SENDER] GStreamer appsink başladı. Yayın: tcp://127.0.0.1:5555 (CTRL+C ile çık)")

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 1_000_000)  # 1s timeout (ns)
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

            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
            # kopya al (buffer bir sonraki frame'de değişebilir)
            frame = frame.copy()
            buf.unmap(mapinfo)

            # JPEG encode (OpenCV yok -> basit turbojpeg yoksa numpy'de kalırız)
            # En pratik: imageio kullanmadan raw gönderelim:
            # Daha hızlı ve basit: raw BGR + header gönderiyoruz
            header = f"{w},{h}".encode()
            sock.send_multipart([b"raw", header, frame.tobytes()])

    except KeyboardInterrupt:
        print("\n[SENDER] Çıkılıyor...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        sock.close()
        ctx.term()
        print("[SENDER] Bitti.")

if __name__ == "__main__":
    main()
