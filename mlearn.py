import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model_path = "wasteclass.h5"
model = tf.keras.models.load_model(model_path)

# Define subcategories
biodegradable_types = ["Food", "Paper", "Cardboard"]
non_biodegradable_types = ["Glass", "Metal", "Plastic"]

# Function to classify subcategory
def classify_subcategory(is_biodegradable, filename):
    filename = filename.lower()
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

# Preprocessing & Prediction Function
def predict_waste(image, filename):
    img = image.resize((224, 224))  
    img = np.array(img).astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)  
    
    prediction = model.predict(img)[0][0]  

    is_biodegradable = prediction <= 0.5
    subcategory = classify_subcategory(is_biodegradable, filename)

    if is_biodegradable:
        return f"Biodegradable - {subcategory} ({(1 - prediction) * 100:.2f}%)"
    else:
        return f"Non-Biodegradable - {subcategory} ({prediction * 100:.2f}%)"

# Streamlit UI
st.title("Waste Classifier")
st.write("Upload an image to classify the waste type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict instantly after upload
    prediction_result = predict_waste(image, uploaded_file.name)
    st.write("Prediction:", prediction_result)
