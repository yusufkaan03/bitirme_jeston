import cv2
import time

def main():
    # V4L2 backend ile aç
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Çözünürlük ve format ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı (V4L2).")
        return

    # FPS tahmini
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("kamera_v4l2_5s.avi", fourcc, fps, (1280, 720))

    print("[INFO] Kayıt başlıyor (5 saniye)...")
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kare okunamadı.")
            break

        out.write(frame)

        if time.time() - start >= 5.0:
            break

    print("[INFO] Kayıt bitti: kamera_v4l2_5s.avi")

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
