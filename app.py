import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from model import MNISTModel

# ---------- Model loading ----------
@st.cache_resource
def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------- Preprocessing (MNIST-style) ----------
def preprocess(canvas_img):
    """
    canvas_img: RGBA numpy array (280x280x4), white strokes on black bg.
    Returns: torch tensor (1,1,28,28), plus the 28x28 preview image.
    """
    # 1. Take one channel, invert to black-digit-on-white
    gray = canvas_img[:, :, 0]
    inverted = 255 - gray

    # 2. Crop to the bounding box of the digit (ignore empty space)
    ys, xs = np.where(inverted > 20)
    if len(xs) == 0:
        return None, None  # nothing drawn
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = inverted[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)

    # 3. Resize the digit to fit inside 20x20 (MNIST standard)
    pil = Image.fromarray(crop)
    w, h = pil.size
    scale = 20.0 / max(w, h)
    pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # 4. Center it on a 28x28 black canvas
    canvas28 = Image.new("L", (28, 28), 0)
    offset = ((28 - pil.size[0]) // 2, (28 - pil.size[1]) // 2)
    canvas28.paste(pil, offset)

    # 5. Normalize to [0,1]
    arr = np.array(canvas28, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()

    return tensor, canvas28


# ---------- UI ----------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognizer")
st.caption("Draw a digit (0–9) in the box below, then click Predict.")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        stroke_width=22,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

with col2:
    st.write("**What the model sees:**")
    preview_slot = st.empty()

if st.button("Predict", type="primary"):
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        tensor, preview = preprocess(canvas_result.image_data)

        if tensor is None:
            st.warning("Canvas is empty — draw something first.")
        else:
            # Show the 28x28 image the model receives
            preview_slot.image(preview, width=140, clamp=True)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())

            st.success(f"Prediction: **{pred}**  (confidence: {conf:.1%})")

            # Top-3 breakdown
            top3 = torch.topk(probs, 3)
            st.write("**Top 3:**")
            for i in range(3):
                digit = int(top3.indices[i].item())
                p = float(top3.values[i].item())
                st.progress(p, text=f"{digit} — {p:.1%}")