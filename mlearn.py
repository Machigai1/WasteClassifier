import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load the trained model dynamically
model_path = "wasteclass.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found! Please ensure 'wasteclass.h5' is in the directory.")
    st.stop()

# Define subcategories manually
biodegradable_types = ["Food", "Paper", "Cardboard"]
non_biodegradable_types = ["Glass", "Metal", "Plastic"]

# Function to determine subcategory
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

# Function to preprocess and predict image
def predict_waste(image, filename):
    img = image.resize((224, 224))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(img)[0][0]  # Get probability for Non-Biodegradable
    
    # Determine category
    is_biodegradable = prediction <= 0.5
    subcategory = classify_subcategory(is_biodegradable, filename)
    
    if is_biodegradable:
        return f"Biodegradable - {subcategory} ({(1 - prediction) * 100:.2f}%)"
    else:
        return f"Non-Biodegradable - {subcategory} ({prediction * 100:.2f}%)"

# Streamlit UI
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")

# Background GIF using CSS
background_gif = "https://i.pinimg.com/originals/97/56/2a/97562a7858d744ef6c30286e35beb9c3.gif"  # Replace with your GIF URL or path

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_gif}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True
)

st.title("♻️ Garbage Classification System")

# Layout with two columns, adjusting height for right column
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.write("Upload an image to classify it as Biodegradable or Non-Biodegradable.")
    st.write("**Biodegradable Types:** Food, Paper, Cardboard")
    st.write("**Non-Biodegradable Types:** Glass, Metal, Plastic")
    
    # Display Limitations
    st.warning("*Limitations:* This model will predict any image as either Biodegradable or Non-Biodegradable, even if the image is not waste-related.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("<div style='min-height:-5px'></div>", unsafe_allow_html=True)  # Increase column height
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=570)  # Reduce image size slightly
        
        # Predict instantly after upload
        prediction_result = predict_waste(image, uploaded_file.name)
        st.write("### Prediction:")
        st.write(prediction_result)
