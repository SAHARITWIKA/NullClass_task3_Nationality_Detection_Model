import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from utils.preprocess import extract_dominant_color, predict_color_name

# Load models
nat_model = load_model("../models/nationality_model.h5")
emo_model = load_model("../models/emotion_model.h5")
age_model = load_model("../models/age_model.h5")

# Label mappings (adjust as per training)
nationality_labels = ['Indian', 'US', 'African', 'Others']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI
st.set_page_config(page_title="Nationality Detection App", layout="centered")
st.title("🧠 Nationality Detection & Conditional Analysis")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image for models
    img_array = np.array(image.resize((64, 64))) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    st.subheader("🔍 Predictions")

    # Predict nationality
    nat_pred = nat_model.predict(img_input)
    nat_index = np.argmax(nat_pred)
    nat_name = nationality_labels[nat_index]
    st.markdown(f"**Nationality:** {nat_name}")

    # Predict emotion
    emo_pred = emo_model.predict(img_input)
    emo_index = np.argmax(emo_pred)
    emo_name = emotion_labels[emo_index]
    st.markdown(f"**Emotion:** {emo_name}")

    # Conditional logic
    if nat_name == 'Indian':
        # Predict age
        age_pred = age_model.predict(img_input)
        age_estimate = int(age_pred[0][0])
        st.markdown(f"**Estimated Age:** {age_estimate}")

        # Predict dress color
        np_image = np.array(image)
        color_rgb = extract_dominant_color(np_image)
        color_name = predict_color_name(color_rgb)
        st.markdown(f"**Dress Color:** {color_name}")

    elif nat_name == 'US':
        # Predict age only
        age_pred = age_model.predict(img_input)
        age_estimate = int(age_pred[0][0])
        st.markdown(f"**Estimated Age:** {age_estimate}")

    elif nat_name == 'African':
        # Predict dress color only
        np_image = np.array(image)
        color_rgb = extract_dominant_color(np_image)
        color_name = predict_color_name(color_rgb)
        st.markdown(f"**Dress Color:** {color_name}")

    else:
        st.info("Only Nationality and Emotion predicted for 'Others'")
