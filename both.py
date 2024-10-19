import streamlit as st
import os
import tempfile
from moviepy.editor import concatenate_videoclips, VideoFileClip
import tensorflow as tf
import numpy as np
import cv2
import pickle

# Directory where sign language videos are stored
SIGN_VIDEO_DIR = 'C:/Users/akogo/Desktop/Sign- Link/videos'  # Update this with your actual path

# Load the trained model for Sign to Text functionality
MODEL_PATH = 'sign_language_recognition_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Load model and label encoder for Sign to Text
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Helper Functions
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
            frames.append(np.array(frame))
        count += 1
    cap.release()
    return np.array(frames)

# Function for predicting sign from video (Sign to Text)
def predict_sign_language(video_path):
    frames = extract_frames_from_video(video_path)
    frames = frames.astype("float32") / 255.0  # Normalize
    predictions = model.predict(frames)
    predicted_label_idx = np.argmax(np.mean(predictions, axis=0))
    predicted_label = le.inverse_transform([predicted_label_idx])[0]
    predicted_label = predicted_label.replace('_', ' ')  # Replace underscores with spaces
    return predicted_label

# Function for Text to Sign conversion (mapping phrases to videos)
def get_video_for_phrase(phrase):
    phrase_cleaned = phrase.replace(' ', '_').lower()
    video_path = os.path.join(SIGN_VIDEO_DIR, f"{phrase_cleaned}.mp4").replace("\\", "/")
    if os.path.exists(video_path):
        return video_path
    else:
        return None

def get_sign_language_video(text):
    phrase = text.strip()
    phrase_with_underscores = phrase.replace(' ', '_')
    video_path = get_video_for_phrase(phrase_with_underscores)
    if video_path:
        return VideoFileClip(video_path)
    else:
        st.warning(f"No sign language video found for phrase: '{phrase}'")
        return None

# Streamlit app layout
st.title("Two-Way Sign Language Communication System")

# Clear button to reset output
clear_output = st.button("Clear Output")
if clear_output:
    # Refresh page to clear the output
    st.experimental_set_query_params()

# User selection for mode of operation
mode = st.selectbox("Select Mode", ("Sign to Text", "Text to Sign"))

if mode == "Sign to Text":
    st.header("Sign to Text")
    # Upload a video for sign recognition
    uploaded_video = st.file_uploader("Upload a video for sign recognition", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Display the uploaded video
        st.video(tfile.name)
        
        # Predict the sign from the uploaded video
        predicted_sign = predict_sign_language(tfile.name)
        st.write(f"Predicted sign: **{predicted_sign}**")

elif mode == "Text to Sign":
    st.header("Text to Sign")
    # Text input for user to enter a phrase
    text_input = st.text_input("Enter a phrase to convert to sign language:")
    
    if text_input:
        st.write(f"Converting: {text_input}")
        
        # Convert the text to a single sign language video
        video_clip = get_sign_language_video(text_input)
        
        if video_clip:
            # Save the video to a temporary file and display it
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_clip.write_videofile(temp_file.name, codec="libx264")
            st.video(temp_file.name)  # Automatically play the video when it's ready
        else:
            st.error("No video found for the given input.")
