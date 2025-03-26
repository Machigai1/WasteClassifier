import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier_model.h5")
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    
    # Convert to RGB if necessary
    if image.shape[-1] == 4:  # Handle RGBA images
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2 or image.shape[-1] == 1:  # Handle grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to match model input size
    image = cv2.resize(image, (150, 150))  # Adjust based on model input shape

    # Normalize
    image = image / 255.0  # Ensure the same normalization as during training

    # Expand dimensions for model prediction
    image = np.expand_dims(image, axis=0)

    return image

# Streamlit UI
st.title("Smart Garbage Classification System")
st.write("Upload an image to classify it as **Biodegradable** or **Non-Biodegradable**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Assuming the model outputs a probability, modify the threshold if necessary
    class_names = ["Biodegradable", "Non-Biodegradable"]
    predicted_class = class_names[int(prediction[0] > 0.5)]

    # Display result
    st.write(f"### Prediction: **{predicted_class}**")
