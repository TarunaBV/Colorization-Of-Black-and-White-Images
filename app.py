import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Image Colorizer", layout="wide")

st.title("🎨 AI Image Colorization")
st.write("Upload a black & white image and get a colorized version")

# ---------------- PATHS ----------------
DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
POINTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    pts = np.load(POINTS_PATH)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

net = load_model()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

    # Colorization
    with st.spinner("Colorizing..."):
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

    with col2:
        st.image(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), caption="Colorized", use_container_width=True)

    # Download
    _, buffer = cv2.imencode('.png', colorized)
    st.download_button("⬇️ Download", buffer.tobytes(), "colorized.png", "image/png")