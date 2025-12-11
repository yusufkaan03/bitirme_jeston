import cv2
import time

def gstreamer_pipeline(width=1280, height=720, fps=30):
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

    print("[INFO] Kamera açıldı: {}x{} @ {} FPS".format(width, height, fps))

    # MJPG codec + AVI container
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("kamera_5s_test.avi", fourcc, fps, (width, height))

    print("[INFO] 5 saniyelik kayıt başlıyor...")
    start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kare okunamadı, çıkılıyor.")
            break

        out.write(frame)
        frame_count += 1

        if time.time() - start >= 5.0:
            break

    print("[INFO] Kayıt bitti. Toplam kare:", frame_count)
    cap.release()
    out.release()
    print("[INFO] Dosya: kamera_5s_test.avi")

if __name__ == "__main__":
    main()
