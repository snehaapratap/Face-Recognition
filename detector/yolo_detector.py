from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='models/yolov8n-face-lindevs.pt'):
        self.model = YOLO(model_path)

    def detect_faces(self, image):
        results = self.model(image)
        bboxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
        return bboxes
