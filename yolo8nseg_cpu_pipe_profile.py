import argparse
import subprocess
import time
import numpy as np
import cv2
from ultralytics import YOLO

def gst_cmd(width, height, fps, flip):
    # BGR raw frames -> stdout (fd=1)
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
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true", help="start with plot enabled")
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--no_window", action="store_true", help="disable imshow (prints stats only)")
    args = ap.parse_args()

    print("[INFO] Loading model:", args.model)
    model = YOLO(args.model)

    frame_bytes = args.w * args.h * 3  # BGR
    cmd = gst_cmd(args.w, args.h, args.fps, args.flip)
    print("[INFO] Starting GStreamer:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )

    draw = args.draw
    t0 = time.perf_counter()
    frames = 0
    fps_s = 0.0
    alpha = 0.1

    # rolling averages
    infer_ms_avg = 0.0
    plot_ms_avg = 0.0
    hud_ms_avg = 0.0
    beta = 0.1

    try:
        print("[INFO] Running (CPU). Keys: q=quit, d=toggle draw")
        while True:
            raw = read_exact(proc.stdout, frame_bytes)
            if raw is None:
                print("[ERROR] GStreamer stream ended.")
                break

            # IMPORTANT: make frame writeable (OpenCV needs it)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.h, args.w, 3)).copy()

            # Inference timing
            t_inf0 = time.perf_counter()
            res = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device="cpu",
                verbose=False
            )
            t_inf1 = time.perf_counter()

            # Plot timing
            t_plot0 = time.perf_counter()
            out = frame
            if draw:
                out = res[0].plot()
            t_plot1 = time.perf_counter()

            # HUD timing (putText)
            t_hud0 = time.perf_counter()
            frames += 1
            dt = time.perf_counter() - t0
            fps_now = frames / dt if dt > 0 else 0.0
            fps_s = fps_now if fps_s == 0 else (1 - alpha) * fps_s + alpha * fps_now

            infer_ms = (t_inf1 - t_inf0) * 1000.0
            plot_ms  = (t_plot1 - t_plot0) * 1000.0

            infer_ms_avg = infer_ms if infer_ms_avg == 0 else (1 - beta) * infer_ms_avg + beta * infer_ms
            plot_ms_avg  = plot_ms  if plot_ms_avg  == 0 else (1 - beta) * plot_ms_avg  + beta * plot_ms

            text1 = f"FPS:{fps_s:.2f} imgsz:{args.imgsz} drawOn:{int(draw)}"
            text2 = f"infer:{infer_ms_avg:.1f}ms plot:{plot_ms_avg:.1f}ms"

            cv2.putText(out, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(out, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            t_hud1 = time.perf_counter()
            hud_ms = (t_hud1 - t_hud0) * 1000.0
            hud_ms_avg = hud_ms if hud_ms_avg == 0 else (1 - beta) * hud_ms_avg + beta * hud_ms

            if args.no_window:
                if frames % 30 == 0:
                    print(f"[STAT] FPS={fps_s:.2f} infer={infer_ms_avg:.1f}ms plot={plot_ms_avg:.1f}ms hud={hud_ms_avg:.2f}ms draw={int(draw)}")
            else:
                cv2.imshow("YOLOv8n-seg CPU (gst pipe) - PROFILE", out)
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
        if not args.no_window:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
