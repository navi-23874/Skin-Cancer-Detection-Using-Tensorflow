import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    layout="wide",
    page_icon="🩺"
)

# Custom CSS for extraordinary styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .card {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }
    h1, h3 {
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
    }
    .stButton button {
        background: linear-gradient(to right, #ff512f, #dd2476);
        border: none;
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #dd2476, #ff512f);
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
    }
    .upload-box {
        border: 2px dashed rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: #ffffff;
        background: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
MODEL_PATH = "skin_cancer_mnist.h5"  
model = load_model(MODEL_PATH)

# Define class labels
class_labels = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Vascular Lesion"
]

# Sidebar for navigation
with st.sidebar:
    st.markdown("<div class='card'><h3>🩺 Skin Cancer Detection Menu</h3></div>", unsafe_allow_html=True)
    selected = st.selectbox(
        "Choose Action", 
        ["Home", "Upload Image"]
    )

# Home Page
if selected == "Home":
    st.markdown("<div class='card'><h3>🏠 Welcome to the Skin Cancer Detection System</h3></div>", unsafe_allow_html=True)
    st.write(
        """
        This advanced AI-based system helps detect skin cancer by analyzing skin lesion images. 
        Leveraging a pre-trained deep learning model, it classifies skin lesions into the following categories:
        
        - **Actinic Keratoses**
        - **Basal Cell Carcinoma**
        - **Benign Keratosis**
        - **Dermatofibroma**
        - **Melanoma**
        - **Nevus**
        - **Vascular Lesion**
        
        Upload an image of a skin lesion to get started.
        """
    )
    st.image("test_image.png")

# Upload Image Page
elif selected == "Upload Image":
    st.markdown("<div class='card'><h3>📤 Upload an Image for Skin Cancer Prediction</h3></div>", unsafe_allow_html=True)

    # Upload Box
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Display the uploaded image
        st.markdown("<div class='upload-box'><h3>Uploaded Image Preview:</h3></div>", unsafe_allow_html=True)
        st.image(uploaded_file, caption="Uploaded Image")

        # Preprocess the image
        image = Image.open(uploaded_file).convert('RGB').resize((28, 28))
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction[0])
        class_name = class_labels[predicted_class]

        # Display the prediction
        st.markdown(f"<div class='card'><h3>🧾 Prediction: {class_name}</h3></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='upload-box'>Drop an image file above to analyze the skin lesion.</div>", unsafe_allow_html=True)
