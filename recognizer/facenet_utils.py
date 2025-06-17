from keras.models import load_model
import numpy as np
import cv2

class FaceNetRecognizer:
    def __init__(self, model_path='models/facenet_keras.h5'):
        self.model = load_model(model_path)
        self.embeddings = {}
    
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
