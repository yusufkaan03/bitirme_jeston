import cv2
import time

def gstreamer_pipeline(
    width=1280,
    height=720,
    fps=30,
):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true"
    )

def main():
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı (GStreamer/OpenCV).")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 30.0

    print(f"[INFO] Kamera açıldı: {width}x{height} @ {fps} FPS")

    # DİKKAT: Codec MJPG, container .avi
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("kamera_5s_mjpg.avi", fourcc, fps, (width, height))

    print("[INFO] Kayıt başlıyor (5 saniye)...")
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kare okunamadı, kayıt durduruluyor.")
            break

        out.write(frame)

        if time.time() - start >= 5.0:
            break

    print("[INFO] Kayıt bitti, dosya kaydedildi: kamera_5s_mjpg.avi")

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
