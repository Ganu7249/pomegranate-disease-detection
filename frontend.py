import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import time

st.set_page_config(page_title='Detect!t',page_icon="./letter-d.png",initial_sidebar_state="auto")

def model_prediction(test_image):
    model = tf.keras.models.load_model("Pomegranate_disease_model.h5",compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)

    return np.argmax(predictions) #return index of max element

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ['Disease Recognition'], 
        icons=['house', 'book','clipboard-data','search'], menu_icon="cast", default_index=0)
    selected

#Main pagerub
if (selected=="Home"): 
    st.header("Pomegranate Fruit Disease Detection Using Deep Learning")
    st.image("./home.jpg")
    st.markdown()

#Prediction
elif(selected=="Disease Recognition"):
    st.header("Disease Recognition")
    st.subheader("Test Your Fruit:")
    test_images = []

    option = st.selectbox('Choose an input Image option:',
                          ('--select option--','Upload', 'Camera'))
    
    if option == "Upload":
        test_images = st.file_uploader("Choose Image(s):", accept_multiple_files=True)
        if(st.button("Show Images")):
            st.image(test_images, width=4, use_column_width=True)

    elif option == "Camera":
        test_images = [st.camera_input("Capture an Image:")]
        if(st.button("Show Images")):
            st.image(test_images, width=4, use_column_width=True)


    if st.button("Predict"):
        for i, test_image in enumerate(test_images):
            st.write(f"Prediction for Image {i + 1}:")
            st.image(test_image, width=4, use_column_width=True)
            result_index = model_prediction(test_image)
            class_name = [
                "Anthracnose",
                "Cercospora",
                "Healthy"
            ]
            predicted_class = class_name[result_index]

            if predicted_class == "Healthy":
                st.success(f"The Fruit is a {predicted_class} Fruit" )
            else:
                st.error(f"The fruit is infected by {predicted_class} Disease")
