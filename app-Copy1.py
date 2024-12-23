import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained DL model
MODEL_PATH = "vgg16_cnn_model.h5"
model = load_model(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(image, target_size):
    """Preprocess the image for model prediction."""
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app layout
st.title("Histopathologic Cancer Detection Model")
st.write("Upload an image to predict whether it's cancerous or not.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","tif"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(96, 96))  

    # Make prediction
    prediction = model.predict(processed_image)
    prediction_label = "Cancerous" if prediction[0][0] > 0.5 else "Non-Cancerous"

    # Display prediction
    st.write(f"Prediction: **{prediction_label}**")
    st.write(f"Confidence: **{prediction[0][0] * 100:.2f}%**")

# Footer
st.write("Developed by Maram")
