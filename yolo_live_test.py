import cv2
from ultralytics import YOLO

# CSI kamera için GStreamer pipeline
# (Jetson Nano / Argus)
PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 max-buffers=1 sync=false"
)

def main():
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı (GStreamer pipeline).")
        return

    # Jetson Nano için genelde en mantıklısı:
    # yolov8n-seg.pt veya yolov8s-seg.pt
    model = YOLO("yolov8n-seg.pt")

    print("[INFO] Canlı segmentasyon başladı. Çıkış: q")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Frame okunamadı.")
            break

        # YOLOv8-seg inference
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        vis = results[0].plot()  # mask + box + label çizilmiş görüntü

        cv2.imshow("YOLOv8-Seg (CSI Live)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
