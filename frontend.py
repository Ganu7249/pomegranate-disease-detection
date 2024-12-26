import streamlit as st
import tensorflow as tf
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "diseasedetectionkey.json"  # Ensure this path is correct

st.set_page_config(page_title='Detect!t', page_icon="./letter-d.png", initial_sidebar_state="auto")

# Google Cloud Storage configuration
BUCKET_NAME = "pomegranatedetectionrecords"  # Your GCS bucket name

# Model file details
model_file_url = "https://drive.google.com/uc?id=1z2STdgv4KQyLhdCDKZmLcBZHlgATa3fj"
model_local_path = "Pomegranate_disease_model.h5"


# Add custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-image: url("data:image/jpeg;base64,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"); /* Replace with actual image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #800000; /* Pomegranate red */
        text-align: center;
    }
    .stButton>button {
        background-color: #800000;
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Function to download the model from Google Drive
def download_model_if_needed():
    if not os.path.exists(model_local_path):
        with st.spinner("Loading model..."):
            r = requests.get(model_file_url, allow_redirects=True)
            open(model_local_path, 'wb').write(r.content)

# Function to upload to Google Cloud Storage
def upload_to_gcs(image_data, filename, prediction):
    try:
        client = storage.Client()  # Uses the environment variable for authentication
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

# Function to predict disease
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_local_path, compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Download the model if not already available
download_model_if_needed()

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
        test_images = st.file_uploader("Choose Image(s):", accept_multiple_files=True)
        if st.button("Show Images"):
            st.image(test_images, width=4, use_column_width=True)

    elif option == "Camera":
        test_images = [st.camera_input("Capture an Image:")]
        if st.button("Show Images"):
            st.image(test_images, width=4, use_column_width=True)

    if st.button("Predict"):
        for i, test_image in enumerate(test_images):
            st.write(f"Prediction for Image {i + 1}:")
            st.image(test_image, width=4, use_column_width=True)

            # Prepare image as byte stream
            image = Image.open(test_image)
            byte_stream = io.BytesIO()
            image.save(byte_stream, format='PNG')
            byte_stream.seek(0)

            # Predict disease
            result_index = model_prediction(test_image)
            class_name = ["Anthracnose", "Cercospora", "Healthy"]
            predicted_class = class_name[result_index]

            # Generate unique filename
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"

            # Upload to GCS
            upload_message = upload_to_gcs(byte_stream, filename, predicted_class)
            st.write(upload_message)

            # Display prediction result
            if predicted_class == "Healthy":
                st.success(f"The Fruit is a {predicted_class} Fruit")
            else:
                st.error(f"The fruit is infected by {predicted_class} Disease")
