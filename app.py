import streamlit as st
import cv2
import numpy as np
from PIL import Image
from recognizer.facenet_utils import FaceNetRecognizer


@st.cache_resource
def load_recognizer():
    recognizer = FaceNetRecognizer()
    recognizer.load_known_faces('data/known_faces')
    return recognizer

recognizer = load_recognizer()

st.title("🔍 Face Recognition App with FaceNet")


uploaded_file = st.file_uploader("Upload an image for face recognition", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = recognizer.recognize(img)
    for box, name in results:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Result", use_column_width=True)
