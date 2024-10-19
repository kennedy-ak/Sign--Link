import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import tempfile
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model and label encoder
model = tf.keras.models.load_model('sign_language_recognition_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Function to extract frames from video
def extract_frames_from_video(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.resize(frame, (128, 128))
            frames.append(img_to_array(frame))
        count += 1
    cap.release()
    return np.array(frames)

# Function to make a prediction on a new video
def predict_sign_language(video_path):
    frames = extract_frames_from_video(video_path)
    frames = frames.astype("float") / 255.0  # Normalize the frames
    predictions = model.predict(frames)
    # Get the most frequent predicted label
    predicted_label_idx = np.argmax(np.mean(predictions, axis=0))
    predicted_label = le.inverse_transform([predicted_label_idx])[0]
    
    # Replace underscores with spaces
    predicted_label = predicted_label.replace('_', ' ')
    
    return predicted_label

# Streamlit app layout
st.title("Sign Language Recognition")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Display the uploaded video
    st.video(tfile.name)
    
    # Predict the sign
    predicted_sign = predict_sign_language(tfile.name)
    
    # Show the result
    st.write(f"Predicted sign: **{predicted_sign}**")
