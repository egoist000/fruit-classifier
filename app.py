# Path: app.py
import streamlit as st
from fastai.vision.all import *

st.title("Fruit Classification")
st.write("This is a simple image classification web app to classify fruits")

# Load your trained model
model = load_learner('fruit-classifier.pkl')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    # Make prediction
    img = PILImage.create(uploaded_file)
    pred, pred_idx, probs = model.predict(img)

    # Show prediction
    st.write("Prediction: ", pred)
    st.write("Probability: ", probs[pred_idx])

    # Show image
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Save image
    img.save('uploaded_image.jpg')