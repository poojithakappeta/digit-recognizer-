import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit below and let the model predict it.")

# Load the trained model
model = load_model("mnist.h5")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict when user clicks button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img  # invert: black digit on white
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first.")
