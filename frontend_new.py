import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import requests
import os
from datetime import datetime
from google.cloud import storage
from PIL import Image
import io

# Load credentials from secrets
credentials_info = st.secrets["google_cloud"]["credentials_json"]

# Write credentials to a temporary file
with open("diseasedetectionkey.json", "w") as temp_file:
    temp_file.write(credentials_info)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "diseasedetectionkey.json"

st.set_page_config(page_title='Detect!t', page_icon="./letter-d.png", initial_sidebar_state="auto")

# Google Cloud Storage configuration
BUCKET_NAME = "pomegranatedetectionrecords"

# FastAPI endpoint
API_URL = "https://<ganeshkende>.huggingface.co/spaces/<pomegranate-disease-detection-app-backend>/predict"

# Function to upload to Google Cloud Storage
def upload_to_gcs(image_data, filename, prediction):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"images/{filename}")

        # Add metadata
        blob.metadata = {
            "prediction": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        blob.upload_from_file(image_data, content_type="image/png")
        return f"Uploaded {filename} successfully to GCS"
    except Exception as e:
        return f"Error uploading to GCS: {e}"

# Function to call FastAPI and get prediction
def call_api_for_prediction(image_file):
    files = {'file': ('image.png', image_file, 'image/png')}
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json().get("predicted_class", "Unknown")
        else:
            return f"API error: {response.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

# Sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ['Disease Recognition'],
                           icons=['house', 'book', 'clipboard-data', 'search'],
                           menu_icon="cast", default_index=0)
    selected

# Main page
if selected == "Disease Recognition":
    st.header("Disease Recognition")
    st.subheader("Test Your Fruit:")
    test_images = []

    option = st.selectbox('Choose an input Image option:',
                          ('--select option--', 'Upload', 'Camera'))

    if option == "Upload":
        test_images = st.file_uploader("Choose Image(s):", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        if st.button("Show Images"):
            st.image(test_images, width=4, use_container_width=True)

    elif option == "Camera":
        test_images = [st.camera_input("Capture an Image:")]
        if st.button("Show Images"):
            st.image(test_images, width=4, use_container_width=True)

    if st.button("Predict"):
        for i, test_image in enumerate(test_images):
            st.write(f"Prediction for Image {i + 1}:")
            st.image(test_image, width=4, use_container_width=True)

            # Open image and prepare byte stream
            image = Image.open(test_image)
            byte_stream = io.BytesIO()
            image.save(byte_stream, format='PNG')
            byte_stream.seek(0)

            # Send image to API
            predicted_class = call_api_for_prediction(byte_stream)
            byte_stream.seek(0)  # Reset stream to beginning for upload

            # Generate unique filename
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"

            # Upload to GCS
            upload_message = upload_to_gcs(byte_stream, filename, predicted_class)
            st.write(upload_message)

            # Display prediction result
            if predicted_class == "Healthy":
                st.success(f"The Fruit is a {predicted_class} Fruit")
            elif predicted_class in ["Anthracnose", "Cercospora"]:
                st.error(f"The fruit is infected by {predicted_class} Disease")
            else:
                st.warning(f"Could not classify the image: {predicted_class}")
