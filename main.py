import cv2
import time

def main():
    # 0: /dev/video0 (ilk kamera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı. /dev/video0 var mı, kamera doğru porta takılı mı kontrol et.")
        return

    # Çözünürlük ve FPS bilgisi al
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Bazı kameralar FPS döndürmüyor, o zaman 30 al
    if fps is None or fps <= 0 or fps > 120:
        fps = 30.0

    print(f"[INFO] Kamera açıldı: {width}x{height} @ {fps} FPS")

    # Video yazıcı ayarı
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # .avi için
    output_path = "kamera_5s_kayit1.avi"

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("[INFO] Kayıt başlıyor (5 saniye)...")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kare okunamadı, kayıt durduruluyor.")
            break

        out.write(frame)

        # 5 saniye doldu mu?
        if time.time() - start_time >= 5.0:
            break

    print("[INFO] Kayıt bitti, dosya kaydedildi:", output_path)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
