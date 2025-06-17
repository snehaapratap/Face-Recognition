from keras.models import load_model
import numpy as np
import cv2

# recognizer/facenet_utils.py
from keras_facenet import FaceNet
import numpy as np

class FaceNetRecognizer:
    def __init__(self):
        self.embedder = FaceNet()
    
    def get_embedding(self, face_img):
        # face_img must be RGB and shaped (160, 160, 3)
        return self.embedder.embeddings([face_img])[0]

    
    def preprocess(self, face):
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        return (face - mean) / std

    def get_embedding(self, face):
        face = self.preprocess(face)
        return self.model.predict(np.expand_dims(face, axis=0))[0]

    def load_known_faces(self, folder_path):
        import os
        for person in os.listdir(folder_path):
            imgs = os.listdir(f"{folder_path}/{person}")
            emb_list = []
            for img_name in imgs:
                img = cv2.imread(f"{folder_path}/{person}/{img_name}")
                emb = self.get_embedding(img)
                emb_list.append(emb)
            self.embeddings[person] = np.mean(emb_list, axis=0)

    def recognize(self, face_img):
        emb = self.get_embedding(face_img)
        min_dist = float('inf')
        identity = 'Unknown'
        for name, db_emb in self.embeddings.items():
            dist = np.linalg.norm(emb - db_emb)
            if dist < 10:  # threshold to tune
                if dist < min_dist:
                    min_dist = dist
                    identity = name
        return identity
