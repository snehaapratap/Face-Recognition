import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

class FaceNetRecognizer:
    def __init__(self):
        self.embedder = FaceNet()
        self.detector = MTCNN()
        self.known_embeddings = []
        self.known_names = []

    def preprocess_face(self, img, box):
        x, y, width, height = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    def load_known_faces(self, folder_path):
        print("[INFO] Loading known faces...")
        for person in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person)
            if not os.path.isdir(person_folder):
                continue  # Skip .DS_Store or non-folder files
            for img_name in os.listdir(person_folder):
                if img_name.startswith('.'):
                    continue  # Skip hidden files
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = self.detector.detect_faces(img)
                if faces:
                    face = self.preprocess_face(img, faces[0]['box'])
                    embedding = self.embedder.embeddings([face])[0]
                    self.known_embeddings.append(embedding)
                    self.known_names.append(person)
        print(f"[INFO] Loaded {len(self.known_names)} known face images.")

    def recognize(self, img):
        results = []
        faces = self.detector.detect_faces(img)
        for face_data in faces:
            box = face_data['box']
            face = self.preprocess_face(img, box)
            embedding = self.embedder.embeddings([face])[0]

            name = "Unknown"
            min_dist = 100
            for known_embedding, known_name in zip(self.known_embeddings, self.known_names):
                dist = np.linalg.norm(known_embedding - embedding)
                if dist < 0.9 and dist < min_dist:
                    name = known_name
                    min_dist = dist

            results.append((box, name))
        return results
  