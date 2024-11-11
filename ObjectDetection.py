import cv2
import os
import gzip
import pickle

# Load classifiers for different detections
classifiers = {
    "face": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_frontalface_default.xml",
    "eye": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_eye_tree_eyeglasses.xml",
    "fullbody": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_fullbody.xml",
    "car": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_car.xml",
    "plate": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_license_plate_rus_16stages.xml",
    "smile": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_smile.xml",
    "right_eye": r"D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_righteye_2splits.xml"
}

# Save classifiers paths
with open('classifiers_paths1.pkl', 'wb') as f:
    pickle.dump(classifiers, f)

# Load classifiers paths
with open('classifiers_paths1.pkl', 'rb') as f:
    loaded_classifiers_paths = pickle.load(f)

# Load classifiers from paths
loaded_classifiers = {key: cv2.CascadeClassifier(path) for key, path in loaded_classifiers_paths.items()}

# Function to detect and draw rectangles
def detect_and_display(image, classifier, color=(0, 255, 0)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

# Main function to process image or video
def process_media(file_path, detection_type):
    if file_path.endswith(('.jpg', '.png')):
        image = cv2.imread(file_path)
        classifier = loaded_classifiers[detection_type.lower()]
        output = detect_and_display(image, classifier, color=(255, 0, 0))
        cv2.imshow(f"{detection_type} Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif file_path.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            classifier = loaded_classifiers[detection_type.lower()]
            output = detect_and_display(frame, classifier, color=(255, 0, 0))
            cv2.imshow(f"{detection_type} Detection", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
file_path = r"D:\pictures\training\happy\img4.jpg"  # Provide path to your image or video file
detection_type = "Face"  # Choose detection type: Face, Eye, Body, Car, License Plate, Smile, Right Eye
process_media(file_path, detection_type)
