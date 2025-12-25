import argparse
import subprocess
import time
import numpy as np
import cv2

import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

def gst_cmd(width, height, fps, flip):
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

def overlay_mask_rcnn_bgr(frame_bgr, pred, score_th=0.5, mask_th=0.5, max_det=10):
    """
    frame_bgr: uint8 (H,W,3)
    pred: dict with boxes, labels, scores, masks (from torchvision Mask R-CNN)
    returns: out_bgr (uint8)
    """
    out = frame_bgr.copy()
    if pred is None:
        return out

    scores = pred["scores"].detach().cpu()
    keep = scores >= score_th
    if keep.sum().item() == 0:
        return out

    # limit detections
    idx = torch.where(keep)[0][:max_det]

    boxes = pred["boxes"][idx].detach().cpu().to(torch.int32)  # (N,4)
    labels = pred["labels"][idx].detach().cpu()                # (N,)
    scores = pred["scores"][idx].detach().cpu()                # (N,)
    masks = pred["masks"][idx].detach().cpu()                  # (N,1,H,W)

    H, W = frame_bgr.shape[:2]
    # boolean masks: (N,H,W)
    mbool = (masks[:, 0] >= mask_th)

    # simple fast overlay: paint mask area green-ish without heavy libs
    # (We avoid fancy alpha blending per-pixel to keep it lighter)
    for i in range(mbool.shape[0]):
        m = mbool[i].numpy()
        if m.any():
            # add a light tint on mask pixels
            out[m, 1] = np.clip(out[m, 1] + 60, 0, 255)  # G channel boost

    # draw boxes + text
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        txt = f"id:{int(labels[i])} {scores[i]:.2f}"
        cv2.putText(out, txt, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--score", type=float, default=0.50, help="score threshold")
    ap.add_argument("--mask_th", type=float, default=0.50, help="mask threshold")
    ap.add_argument("--max_det", type=int, default=10)

    ap.add_argument("--draw", action="store_true", help="start with overlay enabled")
    ap.add_argument("--no_window", action="store_true", help="disable imshow (prints stats only)")
    ap.add_argument("--resize", type=int, default=320, help="resize short side for speed (0=off)")
    args = ap.parse_args()

    print("[INFO] torch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("[INFO] cuda:", torch.cuda.is_available(), "(we will run CPU)")

    # Load model (CPU)
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    model.to("cpu")

    frame_bytes = args.w * args.h * 3
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

    infer_ms_avg = 0.0
    draw_ms_avg = 0.0
    beta = 0.1

    try:
        print("[INFO] Running Mask R-CNN (CPU). Keys: q=quit, d=toggle draw")
        with torch.no_grad():
            while True:
                raw = read_exact(proc.stdout, frame_bytes)
                if raw is None:
                    print("[ERROR] GStreamer stream ended.")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.h, args.w, 3)).copy()

                # Optional resize for speed (keep aspect)
                inp = frame
                if args.resize and args.resize > 0:
                    H, W = frame.shape[:2]
                    short = min(H, W)
                    if short != args.resize:
                        scale = args.resize / float(short)
                        newW = int(round(W * scale))
                        newH = int(round(H * scale))
                        inp = cv2.resize(frame, (newW, newH), interpolation=cv2.INTER_AREA)

                # BGR -> RGB, to tensor [0..1]
                t_inf0 = time.perf_counter()
                rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)
                pred = model([x])[0]
                t_inf1 = time.perf_counter()

                # Draw/overlay
                t_draw0 = time.perf_counter()
                out = frame
                if draw:
                    # If we resized, we need to map predictions back to original frame size
                    if inp is not frame:
                        # scale boxes & masks to original
                        H0, W0 = frame.shape[:2]
                        H1, W1 = inp.shape[:2]
                        sx = W0 / float(W1)
                        sy = H0 / float(H1)

                        pred2 = {}
                        pred2["scores"] = pred["scores"]
                        pred2["labels"] = pred["labels"]

                        boxes = pred["boxes"].clone()
                        boxes[:, [0, 2]] *= sx
                        boxes[:, [1, 3]] *= sy
                        pred2["boxes"] = boxes

                        # masks: (N,1,H1,W1) -> resize to (H0,W0)
                        masks = pred["masks"]  # float
                        if masks.numel() > 0:
                            masks_rs = torch.nn.functional.interpolate(
                                masks, size=(H0, W0), mode="bilinear", align_corners=False
                            )
                        else:
                            masks_rs = masks
                        pred2["masks"] = masks_rs
                        out = overlay_mask_rcnn_bgr(out, pred2, args.score, args.mask_th, args.max_det)
                    else:
                        out = overlay_mask_rcnn_bgr(out, pred, args.score, args.mask_th, args.max_det)
                t_draw1 = time.perf_counter()

                # Stats
                frames += 1
                dt = time.perf_counter() - t0
                fps_now = frames / dt if dt > 0 else 0.0
                fps_s = fps_now if fps_s == 0 else (1 - alpha) * fps_s + alpha * fps_now

                infer_ms = (t_inf1 - t_inf0) * 1000.0
                draw_ms = (t_draw1 - t_draw0) * 1000.0
                infer_ms_avg = infer_ms if infer_ms_avg == 0 else (1 - beta) * infer_ms_avg + beta * infer_ms
                draw_ms_avg  = draw_ms  if draw_ms_avg  == 0 else (1 - beta) * draw_ms_avg  + beta * draw_ms

                cv2.putText(out, f"FPS:{fps_s:.2f} drawOn:{int(draw)} resize:{args.resize}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.putText(out, f"infer:{infer_ms_avg:.1f}ms draw:{draw_ms_avg:.1f}ms score>={args.score}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                if args.no_window:
                    if frames % 30 == 0:
                        print(f"[STAT] FPS={fps_s:.2f} infer={infer_ms_avg:.1f}ms draw={draw_ms_avg:.1f}ms drawOn={int(draw)}")
                else:
                    cv2.imshow("Mask R-CNN R50-FPN v2 CPU (gst pipe) - PROFILE", out)
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
