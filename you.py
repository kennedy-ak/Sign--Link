import streamlit as st
import os
import re
import tempfile
from moviepy.editor import concatenate_videoclips, VideoFileClip

# Directory where sign language videos are stored
SIGN_VIDEO_DIR = './sign_videos'

# Function to find video corresponding to each word
def get_video_for_word(word):
    # Clean the word to remove special characters
    word_cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
    video_path = os.path.join(SIGN_VIDEO_DIR, f"{word_cleaned}.mp4")
    
    if os.path.exists(video_path):
        return video_path
    else:
        return None

# Function to concatenate videos for each word in the input text
def get_sign_language_videos(text):
    words = text.split()
    video_paths = []
    
    for word in words:
        video_path = get_video_for_word(word)
        if video_path:
            video_paths.append(VideoFileClip(video_path))
        else:
            st.warning(f"No sign language video found for word: {word}")
    
    if video_paths:
        # Concatenate all the videos
        final_clip = concatenate_videoclips(video_paths, method="compose")
        # Save the concatenated video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        final_clip.write_videofile(temp_file.name, codec="libx264")
        return temp_file.name
    else:
        return None

# Streamlit app layout
st.title("Text-to-Sign Language Converter")

# Text input for user to enter text
text_input = st.text_input("Enter text to convert to sign language")

if text_input:
    st.write(f"Converting: {text_input}")
    
    # Convert the text to a sequence of sign language videos
    video_file = get_sign_language_videos(text_input)
    
    if video_file:
        # Display the concatenated video
        st.video(video_file)
    else:
        st.error("No videos found for the given input.")
