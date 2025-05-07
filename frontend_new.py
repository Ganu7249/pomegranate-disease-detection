import streamlit as st
import numpy as np
import requests
import os
from datetime import datetime
from google.cloud import storage
from PIL import Image
import io
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import tensorflow as tf

# Load credentials from secrets
credentials_info = st.secrets["google_cloud"]["credentials_json"]
with open("diseasedetectionkey.json", "w") as temp_file:
    temp_file.write(credentials_info)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "diseasedetectionkey.json"

st.set_page_config(page_title='Detect!t', page_icon="./letter-d.png", initial_sidebar_state="auto")

# Google Cloud Storage configuration
BUCKET_NAME = "pomegranatedetectionrecords"

# Remote file URLs
drive_files = {
    "model": "https://drive.google.com/uc?export=download&id=11b4eYmu5T-hdmElE-WgjlOhmagP1GFSZ",
    "detectron_weights": "https://drive.google.com/uc?export=download&id=1-G9YqfpkFp4EJuNzwp3Tp2OUVAv7r2f0",
    "selected_indices": "https://drive.google.com/uc?export=download&id=1-Wo2bbPvv5S3_btykA4czQIOc0RVd2MH"
}

# Download required files if not present
def download_model_if_needed():
    files = {
        "model_1.h5": drive_files["model"],
        "model_final.pth": drive_files["detectron_weights"],
        "selected_indices.npy": drive_files["selected_indices"]
    }
    for file_name, url in files.items():
        if not os.path.exists(file_name):
            r = requests.get(url, allow_redirects=True)
            open(file_name, 'wb').write(r.content)

download_model_if_needed()

# Load all models
cnn_hboa_model = tf.keras.models.load_model("model_1.h5")
selected_indices = np.load("selected_indices.npy")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Image processing and prediction logic
def segment_image(image_np):
    outputs = predictor(image_np)
    if len(outputs["instances"].pred_masks) == 0:
        return None
    mask = outputs["instances"].pred_masks[0].cpu().numpy()
    return cv2.bitwise_and(image_np, image_np, mask=mask.astype(np.uint8))

def extract_features(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features.squeeze().numpy().flatten()

def select_hboa_features(features):
    return features[selected_indices[:1032]].reshape(1, -1)

def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    segmented = segment_image(image_bgr)
    if segmented is None:
        return "Segmentation failed", None
    segmented_pil = Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    features = extract_features(segmented_pil)
    selected_features = select_hboa_features(features)
    predictions = cnn_hboa_model.predict(selected_features)
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    class_labels = ["Anthracnose", "Bacterial Blight", "Cercospora", "Healthy"]
    return f"{class_labels[predicted_class]} ({confidence:.2f})", predicted_class

# Upload image and prediction to GCS
def upload_to_gcs(image_data, filename, prediction):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"images/{filename}")
        blob.metadata = {
            "prediction": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        blob.upload_from_file(image_data, content_type="image/png")
        return f"Uploaded {filename} successfully to GCS"
    except Exception as e:
        return f"Error uploading to GCS: {e}"

# Sidebar UI
with st.sidebar:
    selected = option_menu("Main Menu", ['Disease Recognition'], 
                           icons=['house'], menu_icon="cast", default_index=0)

if selected == "Disease Recognition":
    st.header("Disease Recognition")
    st.subheader("Test Your Fruit:")

    test_images = []
    option = st.selectbox('Choose an input Image option:', ('--select option--', 'Upload', 'Camera'))

    if option == "Upload":
        test_images = st.file_uploader("Choose Image(s):", accept_multiple_files=True)
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

            image = Image.open(test_image)
            byte_stream = io.BytesIO()
            image.save(byte_stream, format='PNG')
            byte_stream.seek(0)

            result_text, result_index = model_prediction(test_image)
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
            upload_message = upload_to_gcs(byte_stream, filename, result_text)
            st.write(upload_message)

            if "Healthy" in result_text:
                st.success(f"The Fruit is a {result_text}")
            elif "Segmentation failed" in result_text:
                st.warning(result_text)
            else:
                st.error(f"The fruit is infected by {result_text}")
