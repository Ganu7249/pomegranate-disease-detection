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
        background: url('https://media.istockphoto.com/id/502425210/photo/ripe-pomegranates-on-tree.jpg?s=612x612&w=0&k=20&c=tUtnExISwNAsuGSjBEqthukhX1EoU8Dmvb03Df5WeZw=') no-repeat center center fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        background: url('https://i.pinimg.com/736x/63/26/5c/63265c0085bc582df5798bc5f91c0824.jpg') no-repeat center center fixed;
        background-size: cover;
    }

    /* Transparent overlay for sidebar content */
    [data-testid="stSidebar"] > div:first-child {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
        filter: blur(8px);
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.6); /* Light overlay for text readability */
        z-index: 1;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 1.0);
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
    /* Text Styling in Sidebar */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .css-1cpxqw2 {
        color: #800000; /* Pomegranate red for headings */
        font-weight: bold;
    }

    /* Buttons in Sidebar */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #800000; /* Pomegranate red background */
        color: white; /* White text for better visibility */
        font-size: 16px;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #a00000; /* Slightly darker red on hover */
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
    st.header("Welcome to ADCET AgroCare ")
    st.subheader("Test Your Pomegranate:")
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
