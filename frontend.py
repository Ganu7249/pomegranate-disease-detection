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
