import time
import argparse
import cv2
from ultralytics import YOLO

def gst_csi(width=1280, height=720, fps=30, flip=0):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0", help="0=cuda, cpu=cpu")
    ap.add_argument("--draw", action="store_true", help="Start with drawing enabled")
    args = ap.parse_args()

    print("[INFO] Loading model:", args.model)
    model = YOLO(args.model)

    pipe = gst_csi(args.w, args.h, args.fps, flip=0)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] OpenCV cannot open camera with CAP_GSTREAMER.")
        print("        -> If GStreamer enabled is False in cv2 build, use system OpenCV (JetPack) or uninstall pip opencv.")
        return

    draw = args.draw
    t0 = time.perf_counter()
    frames = 0
    fps_s = 0.0
    alpha = 0.1

    print("[INFO] Running. Keys: q=quit, d=toggle draw")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        t_inf0 = time.perf_counter()
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        t_inf1 = time.perf_counter()

        out = frame
        if draw:
            out = res[0].plot()

        frames += 1
        dt = time.perf_counter() - t0
        fps_now = frames / dt if dt > 0 else 0.0
        fps_s = fps_now if fps_s == 0 else (1 - alpha) * fps_s + alpha * fps_now

        inf_ms = (t_inf1 - t_inf0) * 1000.0
        cv2.putText(out, f"FPS: {fps_s:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(out, f"infer: {inf_ms:.1f} ms  draw:{int(draw)}  imgsz:{args.imgsz}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Jetson YOLOv8n-seg (CSI)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("d"):
            draw = not draw

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
