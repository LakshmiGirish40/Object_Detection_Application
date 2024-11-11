import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import pickle


# Load pre-trained classifiers from the pickle file
def load_classifiers(file_path):
    with open(file_path, 'rb') as f:
        classifier_paths = pickle.load(f)
    
    classifiers = {}
    for key, path in classifier_paths.items():
        classifiers[key] = cv2.CascadeClassifier(path)
    
    return classifiers

# Load the classifiers (make sure to point to your actual pickle file)
loaded_classifiers = load_classifiers("classifiers_paths.pkl")

# Title of the Streamlit app
st.title("OpenCV Object Detection App")

# Sidebar for user selection
st.sidebar.title("Detection Options")
app_mode = st.sidebar.selectbox("Choose Detection Type", [
    "Face Detection",
    "Eye Detection",
    "Body Detection",
    "Car Detection",
    "License Plate Detection",
    "Smile Detection",
    "Right Eye Detection"
])


# Option to select a predefined media file or upload a new one
file_option = st.sidebar.radio("Choose a file to process", ["Upload your own file"])
if file_option == "Upload your own file":
    uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "png", "mp4"])
    uploaded_file_path = uploaded_file

# Process the uploaded or predefined file
if file_option == "Upload your own file" and uploaded_file is not None:
    # Handle file upload (image or video)
    if uploaded_file.name.endswith(('.jpg', '.png')):
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if app_mode == "Face Detection":
            # Detect Faces
           classifier = loaded_classifiers["face"]  # or another classifier type
           faces = classifier.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Video Frame", use_column_width=True)
            time.sleep(0.1)  # Add a small delay to allow real-time video processing

        elif app_mode == "Eye Detection":
            # Detect Eyes
            eyes = loaded_classifiers['eye'].detectMultiScale(gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Eye Detection", use_column_width=True)

        elif app_mode == "Body Detection":
            # Detect Body (Full Body)
            bodies =loaded_classifiers['full_body'].detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in bodies:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Body Detection", use_column_width=True)

        elif app_mode == "Car Detection":
            # Detect Cars
            cars = loaded_classifiers['car'].detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in cars:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Car Detection", use_column_width=True)
            
       
        elif app_mode == "License Plate Detection":
            # Detect License Plates
            plates = loaded_classifiers['plate'].detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in plates:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="License Plate Detection", use_column_width=True)
            
        elif app_mode == "Smile Detection":
            # Detect Faces first
            faces = loaded_classifiers['face'].detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
                # ROI for smile detection within the detected face
                roi_gray = gray[y:y + h, x:x + w]
                smiles = loaded_classifiers['smile'].detectMultiScale(roi_gray, 1.8, 20)  # Adjust these parameters as needed
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)  # Green box for smiles
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Right Eye Detection", use_column_width=True)
                time.sleep(0.1)
  
        elif app_mode == "Right Eye Detection": 
            faces =loaded_classifiers['face'].detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces: 
                cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
                    
                # Define regions of interest (ROI) for eyes within each face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                
                # Detect eyes within the face region
                reyes =loaded_classifiers['right_eye'].detectMultiScale(roi_gray, 1.3, 5)
                for (ex, ey, ew, eh) in reyes: 
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Right Eye Detection", use_column_width=True)
                

        elif app_mode == "Eye Detection": 
            for (x, y, w, h) in faces: 
                cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
                    
                # Define regions of interest (ROI) for eyes within each face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                    
                # Detect eyes within the face region
                eyes = loaded_classifiers['eye'].detectMultiScale(roi_gray, 1.3, 5)
                for (ex, ey, ew, eh) in eyes: 
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Eye Detection", use_column_width=True)


    # If it's a video
    elif uploaded_file.name.endswith('.mp4'):
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Process Video
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if app_mode == "Face Detection":
                faces = loaded_classifiers['face'].detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
                # Display the processed frame in Streamlit
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")

            elif app_mode == "Eye Detection":
                eyes = loaded_classifiers['eye'].detectMultiScale(gray, 1.3, 5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")
                 
            elif app_mode == "Body Detection":
                bodies = loaded_classifiers['full_body'].detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")
                
            elif app_mode == "Smile Detection":
                # Detect Body (Full Body)
                smile =loaded_classifiers['smile'].detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in smile:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")


            elif app_mode == "Car Detection":
                cars =loaded_classifiers['car'].detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in cars:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")
            
            
            elif app_mode == "License Plate Detection":
                plates =loaded_classifiers['plate'].detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in plates:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")
                    
            elif app_mode == "Right Eye Detection": 
                reyes = loaded_classifiers['right_eye'].detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in reyes:
                      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                st.image(frame, caption="Video Frame", use_column_width=True, channels="BGR")      

            # Display video frame
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Video Frame", use_column_width=True)

        cap.release()
        # Remove the temporary file after processing
        os.remove(video_path)

else:
    st.write("Please upload an image or video file to get started.")
