# Object_Detection_Application


Project Title:
OpenCV-Based Object Detection App Using HaarCascade Classifiers

Overview:
Developed a versatile object detection application using OpenCV and HaarCascade Classifiers to detect faces, eyes, full bodies, cars, license plates, smiles, and specific facial features (e.g., right eye) in both images and video files. This app is suitable for a range of applications, from facial recognition systems to surveillance and automotive object detection.

Objectives:
Implement a scalable solution for object detection using OpenCV’s HaarCascade Classifiers.
Enable real-time detection on both images and video files, with visual feedback through bounding boxes on detected objects.
Provide flexibility to detect various objects by choosing the desired classifier.
Technical Stack:
Programming Language: Python
Libraries and Frameworks: OpenCV, Pickle, HaarCascade Classifiers
File Formats Supported: JPEG, PNG (for images), MP4 (for video files)
Components of the Project:
Classifiers Management:
Components of the Project:
1 . Classifiers Management:
Description: Loaded multiple Haar Cascade XML files for different types of object detection (e.g., face, eye, full body, car, etc.) and saved their paths in a dictionary.
Storage: Used pickle to save and load classifier paths in a .pkl file for easier management and retrieval, enabling efficient access to each specific classifier type.
Image Preprocessing: Each frame is converted to grayscale to simplify the object detection process.
Detection Function: Implemented a detect_and_display function that takes an image and a classifier, applies the classifier to detect objects, and draws bounding boxes around detected regions.
Real-Time Detection: For video input, frames are processed in real-time, with detection results displayed in a continuous loop until the user quits.
2. Media Processing Functionality:
Image Mode: For image files (e.g., .jpg, .png), loads the image, applies the specified detection type, and displays the detection result in a pop-up window.
Video Mode: For video files (e.g., .mp4), reads each frame, applies detection, and displays the detection result in real-time. The user can exit by pressing 'q'.
3. User Interaction:
Input Parameters: Users specify the path to the media file and the type of object to detect (e.g., "Face," "Eye," "Body," etc.).
Visual Feedback: Detected objects are highlighted with colored bounding boxes, providing a clear visual indication of detected regions.
Challenges Addressed:
Classifier Management: Simplified the management of multiple classifiers by organizing them in a dictionary and saving paths for easy loading.
Real-Time Processing: Implemented efficient frame processing for real-time detection in video files.
User Flexibility: Created a dynamic framework where users can specify different detection types based on the chosen classifier.
Results:
The application successfully detects and highlights various objects with bounding boxes across different media formats, providing a functional base for real-time object detection.
Potential Applications:
Facial recognition systems
Driver assistance and surveillance systems
Emotion detection and analysis in video frames
Feature-specific image analysis and tagging
