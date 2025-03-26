import streamlit as st
 import tensorflow as tf
 import numpy as np
 import cv2
 import os
 from PIL import Image
 
 # Load the trained model dynamically
 MODEL_PATH = "wasteclass.h5"
 model = tf.keras.models.load_model(MODEL_PATH)
 
 # Define subcategories manually
 biodegradable_types = ["Food", "Paper", "Cardboard"]
 non_biodegradable_types = ["Glass", "Metal", "Plastic"]
 
 # Function to determine subcategory (Modify logic if needed)
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
 def predict_waste(image):
     img = np.array(image)
     img = cv2.resize(img, (224, 224))  # Resize to match model input size
     img = img / 255.0  # Normalize
     img = np.expand_dims(img, axis=0)  # Add batch dimension
     
     prediction = model.predict(img)[0][0]  # Get probability for Non-Biodegradable
     is_biodegradable = prediction <= 0.5
     subcategory = classify_subcategory(is_biodegradable, "uploaded_image")
     
     if is_biodegradable:
         return f"Biodegradable - {subcategory} ({(1 - prediction) * 100:.2f}%)"
     else:
         return f"Non-Biodegradable - {subcategory} ({prediction * 100:.2f}%)"
 
 # Streamlit UI
 st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")
 st.title("♻️ Waste Classification System")
 st.write("Upload an image to classify it as Biodegradable or Non-Biodegradable.")
 st.write("Biodegradable Types: Food, Paper, Cardboard")
 st.write("Non-Biodegradable Types: Glass, Metal, Plastic")
 
 # Display Limitations
 st.warning("**Limitations:** This model will predict any image as either Biodegradable or Non-Biodegradable, even if the image is not waste-related.")
 
 # File uploader
 uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
 if uploaded_file is not None:
     image = Image.open(uploaded_file)
     st.image(image, caption="Uploaded Image", use_container_width=True)
     
     # Perform prediction
     result = predict_waste(image)
     
     # Display result
     st.success(result)
