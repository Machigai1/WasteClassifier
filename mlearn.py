import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "wasteclass.h5")
model = tf.keras.models.load_model(model_path)

# Define subcategories manually
biodegradable_types = ["Food", "Paper", "Cardboard"]
non_biodegradable_types = ["Glass", "Metal", "Plastic"]

# Function to determine subcategory
def classify_subcategory(is_biodegradable, image_name):
    filename = image_name.lower()
    if is_biodegradable:
        if "paper" in filename:
            return "Paper"
        elif "cardboard" in filename:
            return "Cardboard"
        else:
            return "Food"
    else:
        if "glass" in filename:
            return "Glass"
        elif "metal" in filename:
            return "Metal"
        else:
            return "Plastic"

# Function to preprocess and predict image
def predict_waste(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)[0][0]
    is_biodegradable = prediction <= 0.5
    subcategory = classify_subcategory(is_biodegradable, "uploaded_image")
    
    if is_biodegradable:
        return f"Biodegradable - {subcategory} ({(1 - prediction) * 100:.2f}%)"
    else:
        return f"Non-Biodegradable - {subcategory} ({prediction * 100:.2f}%)"

# Streamlit UI
st.title("Waste Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        prediction_result = predict_waste(image)
        st.success(f"Prediction: {prediction_result}")
