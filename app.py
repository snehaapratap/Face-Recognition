import streamlit as st
import numpy as np
from PIL import Image
from recognizer.facenet_utils import FaceNetRecognizer

# ✅ Set page config FIRST
st.set_page_config(page_title="Face Recognition", layout="centered")

# App Title
st.title("Face Recognition using FaceNet")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Initialize recognizer
recognizer = FaceNetRecognizer()
recognizer.load_known_faces("data/known_faces")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run recognition
    result_image = recognizer.recognize_faces(np.array(image))

    st.image(result_image, caption="Recognition Result", use_column_width=True)
