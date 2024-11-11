import pickle
import cv2

# Paths to the XML files for the classifiers
classifier_paths = {
    "face": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_frontalface_default.xml",
    "eye": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_eye_tree_eyeglasses.xml",
    "fullbody": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_fullbody.xml",
    "car": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_car.xml",
    "plate": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_license_plate_rus_16stages.xml",
    "smile": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_smile.xml",
    "right_eye": "D:/Data_Science&AI/Spyder/OpenCV/Haarcascades/haarcascade_righteye_2splits.xml"
}

# Save the paths using pickle
with open("classifier_paths.pkl", "wb") as f:
    pickle.dump(classifier_paths, f)

# Later, to load the classifiers
with open("classifier_paths.pkl", "rb") as f:
    loaded_paths = pickle.load(f)
  

# Re-initialize the classifiers from loaded paths
classifiers = {
    key: cv2.CascadeClassifier(path) for key, path in loaded_paths.items()
}

import os
os.getcwd()
# Now `classifiers` contains the reloaded CascadeClassifier objects
