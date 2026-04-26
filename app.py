import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io
import urllib.request

def download_file(url, dest):
    if not os.path.exists(dest):
        try:
            with st.spinner(f"Downloading {os.path.basename(dest)}..."):
                urllib.request.urlretrieve(url, dest)
        except Exception as e:
            st.error("❌ Failed to download model. Please try again later.")
            st.stop()

# Page config
st.set_page_config(page_title="Image Colorization", layout="centered")

st.title("🎨 Black & White Image Colorization")
st.write("Upload a grayscale image and see it come to life with colors!")

# Use relative path (IMPORTANT for portability)
DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_URL = "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_release_v2.caffemodel"

PROTOTXT_URL = "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"

POINTS_URL = "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy"

MODEL_PATH = os.path.join(DIR, "model/colorization_release_v2.caffemodel")
PROTOTXT_PATH = os.path.join(DIR, "model/colorization_deploy_v2.prototxt")
POINTS_PATH = os.path.join(DIR, "model/pts_in_hull.npy")

os.makedirs(os.path.join(DIR, "model"), exist_ok=True)

download_file(MODEL_URL, MODEL_PATH)
download_file(PROTOTXT_URL, PROTOTXT_PATH)
download_file(POINTS_URL, POINTS_PATH)


PROTOTXT = os.path.join(DIR, "model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model/pts_in_hull.npy")
MODEL = os.path.join(DIR, "model/colorization_release_v2.caffemodel")

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

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Preprocess
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorization
    st.write("Colorizing... ⏳")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    st.subheader("Colorized Image")
    st.image(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), use_container_width=True)

    _, buffer = cv2.imencode('.png', colorized)
    byte_io = io.BytesIO(buffer)

    st.download_button(
        label="⬇️ Download Image",
        data=byte_io,
        file_name="colorized.png",
        mime="image/png",
        use_container_width=True
    )
