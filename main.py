import os
import cv2
from recognizer.facenet_utils import FaceNetRecognizer

def main():
    # Initialize FaceNet Recognizer
    recognizer = FaceNetRecognizer()

    # Load known faces from folder
    recognizer.load_known_faces('data/known_faces')

    # Path to test images
    test_folder = 'data/test_images'

    # Process each test image
    for img_name in os.listdir(test_folder):
        if img_name.startswith('.'):
            continue  # Skip system files like .DS_Store

        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Unable to read {img_path}")
            continue

        # Recognize faces in the image
        results = recognizer.recognize(img)

        # Draw bounding boxes and labels
        for box, name in results:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show result
        cv2.imshow(f"Face Recognition - {img_name}", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
