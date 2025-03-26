import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model dynamically
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("wasteclass.h5")

model = load_model()

# Define class labels
class_labels = [
    "Biodegradable - Food", "Biodegradable - Paper", "Biodegradable - Cardboard",
    "Non-Biodegradable - Glass", "Non-Biodegradable - Metal", "Non-Biodegradable - Plastic"
]

# Function to preprocess and predict image
def predict_waste(image):
    img = np.array(image.convert("RGB"))  # Ensure RGB format
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)  # Get model predictions
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    confidence = np.max(prediction) * 100  # Confidence percentage
    
    return f"{class_labels[predicted_class]} ({confidence:.2f}%)"

# Streamlit UI
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Waste Classification System")
st.write("Upload an image to classify it as Biodegradable or Non-Biodegradable.")

# Display Class Categories
st.markdown("**Biodegradable:** Food, Paper, Cardboard")
st.markdown("**Non-Biodegradable:** Glass, Metal, Plastic")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    result = predict_waste(image)
    
    # Display result
    st.success(result)
