import cv2
from detector.yolo_detector import YOLODetector
from recognizer.facenet_utils import FaceNetRecognizer

detector = YOLODetector()
recognizer = FaceNetRecognizer()
recognizer.load_known_faces('data/known_faces')

img = cv2.imread('data/test_images/group.jpg')
bboxes = detector.detect_faces(img)

for (x1, y1, x2, y2) in bboxes:
    face = img[y1:y2, x1:x2]
    name = recognizer.recognize(face)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
