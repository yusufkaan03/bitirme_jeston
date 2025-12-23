import argparse
import subprocess
import time
import numpy as np
import cv2
from ultralytics import YOLO

def gst_cmd(width, height, fps, flip):
    # BGR raw frames -> stdout (fd=1)
    # queue leaky=downstream keeps latency low
    pipeline = (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "fdsink fd=1 sync=false"
    )
    return ["gst-launch-1.0", "-q"] + pipeline.split()

def read_exact(stream, nbytes):
    buf = b""
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--flip", type=int, default=0)
    args = ap.parse_args()

    print("[INFO] Loading model:", args.model)
    model = YOLO(args.model)

    frame_bytes = args.w * args.h * 3  # BGR
    cmd = gst_cmd(args.w, args.h, args.fps, args.flip)
    print("[INFO] Starting GStreamer:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,   # istersen PIPE yapıp debug alabilirsin
        bufsize=0
    )

    draw = args.draw
    t0 = time.perf_counter()
    frames = 0
    fps_s = 0.0
    alpha = 0.1

    try:
        print("[INFO] Running (CPU). Keys: q=quit, d=toggle draw")
        while True:
            raw = read_exact(proc.stdout, frame_bytes)
            if raw is None:
                print("[ERROR] GStreamer stream ended.")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.h, args.w, 3))

            t_inf0 = time.perf_counter()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)
            t_inf1 = time.perf_counter()

            out = frame
            if draw:
                out = res[0].plot()

            frames += 1
            dt = time.perf_counter() - t0
            fps_now = frames / dt if dt > 0 else 0.0
            fps_s = fps_now if fps_s == 0 else (1 - alpha) * fps_s + alpha * fps_now
            inf_ms = (t_inf1 - t_inf0) * 1000.0

            cv2.putText(out, f"FPS: {fps_s:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(out, f"infer: {inf_ms:.1f} ms  draw:{int(draw)} imgsz:{args.imgsz}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("YOLOv8n-seg CPU (gst pipe)", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("d"):
                draw = not draw

    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
