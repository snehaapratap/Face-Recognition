import streamlit as st
import numpy as np
import cv2
from PIL import Image
from recognizer.facenet_utils import FaceNetRecognizer

# Initialize FaceNet Recognizer
@st.cache_resource
def load_recognizer():
    recognizer = FaceNetRecognizer()
    recognizer.load_known_faces('data/known_faces')
    return recognizer

recognizer = load_recognizer()

# App UI
st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("👤 Face Recognition App")
st.write("Upload an image and the model will identify known faces using FaceNet.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Recognize faces
    results = recognizer.recognize(img_bgr)

    # Draw bounding boxes and names
    for box, name in results:
        x, y, w, h = box
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_np, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(img_np, caption="Recognized Faces", use_column_width=True)
