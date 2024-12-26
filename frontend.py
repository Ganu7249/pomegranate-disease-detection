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
        background-color: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        /*background: url('data:image/jpeg;base64,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') no-repeat center center fixed;*/
        background-size: cover;
    }

    /* Transparent overlay for sidebar content */
    [data-testid="stSidebar"] > div:first-child {
        border-radius: 10px;
        padding: 10px;
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.5); /* Light overlay for text readability */
        z-index: 1;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 1.9);
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
        font-weight: bold;
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
        font-weight: bold;
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
