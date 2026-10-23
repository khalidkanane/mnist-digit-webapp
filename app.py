import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image
from model import MNISTModel  # Your model class

@st.cache_resource
def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("Handwritten Digit Recognizer")
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    if st.button("Predict"):
        # Preprocessing: grayscale, resize to 28x28, normalize
        img = canvas_result.image_data.astype('uint8')
        img_gray = Image.fromarray(img).convert('L').resize((28, 28))
        arr = np.array(img_gray) / 255.0
        arr = 1.0 - arr  # invert: white strokes on black -> black digit on white
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            pred = model(tensor).argmax().item()
        st.success(f"Prediction: {pred}")