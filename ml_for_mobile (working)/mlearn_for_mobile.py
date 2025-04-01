import tensorflow as tf
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# 🔹 Get absolute directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Load the trained model dynamically
model_path = os.path.join(BASE_DIR, "wasteclass.h5")
model = tf.keras.models.load_model(model_path)

# Define subcategories manually
biodegradable_types = ["Food", "Paper", "Cardboard"]
non_biodegradable_types = ["Glass", "Metal", "Plastic"]

# Function to determine subcategory (Mock logic: Modify as needed)
def classify_subcategory(is_biodegradable, image_path):
    filename = os.path.basename(image_path).lower()
    
    # Basic keyword matching (Modify this logic if needed)
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
def predict_waste(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(img)[0][0]  # Get probability for Non-Biodegradable
    
    # Determine category
    is_biodegradable = prediction <= 0.5
    subcategory = classify_subcategory(is_biodegradable, image_path)
    
    if is_biodegradable:
        return f"Biodegradable - {subcategory} ({(1 - prediction) * 100:.2f}%)"
    else:
        return f"Non-Biodegradable - {subcategory} ({prediction * 100:.2f}%)"

# Function to open file dialog and classify image
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load image for UI
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep reference

        # Get prediction
        result = predict_waste(file_path)
        result_label.config(text="Prediction: " + result, font=("Arial", 14, "bold"))

# Create UI window
root = tk.Tk()
root.title("Waste Classifier")
root.geometry("450x600")
root.resizable(False, False)  # 🔹 Disable fullscreen/maximizing

# 🔹 Load and set background image dynamically
bg_image_path = os.path.join(BASE_DIR, "background.jpg")
if os.path.exists(bg_image_path):
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((450, 600))  # Resize to match window size
    bg_photo = ImageTk.PhotoImage(bg_image)

    bg_label = tk.Label(root, image=bg_photo)  # Set image as background
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover full window

# Upload Button (Classify Button)
upload_button = tk.Button(root, text="Classify", command=upload_and_predict, font=("Arial", 14, "bold"), 
                          bg="#226622", fg="white", padx=15, pady=10, borderwidth=2, 
                          relief="ridge", cursor="hand2")  # Darker green for contrast
upload_button.pack(pady=10)

# Image Display
image_label = tk.Label(root, bg="green")
image_label.pack()

# Prediction Result
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16, "bold"), fg="black", bg="green")
result_label.pack(pady=20)

# Run the UI
root.mainloop()
