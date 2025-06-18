# 🎭 Face Recognition System using Facenet and YOLOV8 

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![FaceNet](https://img.shields.io/badge/FaceNet-Keras-orange)
![YOLO](https://img.shields.io/badge/YOLO-v8-green)

A powerful and user-friendly face recognition system that combines the accuracy of FaceNet with the speed of YOLO for real-time face detection and recognition.

</div>

## ✨ Features

- 🔍 Real-time face detection using YOLOv8
- 👤 Face recognition powered by FaceNet
- 🖼️ Support for multiple image formats (JPG, JPEG, PNG)
- 🌐 User-friendly web interface built with Streamlit
- 📊 Batch processing capability for multiple images
- 🎯 High accuracy face matching with customizable threshold

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/snehaapratap/Face-Recognition.git
   cd Face-Recognition
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Face-Recognition/
├── app.py              # Streamlit web application
├── main.py            # Command-line interface
├── detector/          # Face detection modules
├── recognizer/        # Face recognition modules
├── data/             # Training and test data
│   ├── known_faces/  # Known face images
│   └── test_images/  # Test images
└── models/           # Pre-trained models
```

## 🛠️ Usage

### Web Interface
1. Launch the Streamlit app
2. Upload an image containing faces
3. View the recognition results with bounding boxes and labels

### Command Line
```bash
python main.py
```

## 📝 Requirements

- Python 3.8+
- OpenCV
- TensorFlow
- Keras-FaceNet
- Streamlit
- MTCNN
- Ultralytics YOLO

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


