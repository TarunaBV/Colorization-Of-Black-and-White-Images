import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import gdown
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎨 AI Image Colorization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Transform black & white images into color using Deep Learning</p>", unsafe_allow_html=True)

st.info("⚡ First run will download AI model (~100MB). Please wait...")

# ---------------- PATH SETUP ----------------
DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
POINTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# ---------------- GOOGLE DRIVE FILE IDs ----------------
# 👉 Replace this with YOUR OWN FILE ID
MODEL_FILE_ID = "1AjPyRraTHeJusyjY6800smc3_bRT2Iaw"

# ---------------- DOWNLOAD FUNCTION ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model (~100MB)..."):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

def download_others():
    import urllib.request

    if not os.path.exists(PROTOTXT_PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt",
            PROTOTXT_PATH
        )

    if not os.path.exists(POINTS_PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
            POINTS_PATH
        )

# ---------------- DOWNLOAD FILES ----------------
download_model()
download_others()

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

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload a Black & White Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mobile_view = st.checkbox("📱 Mobile View (stack images)")

    if mobile_view:
        st.subheader("🖤 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ---------------- COLORIZATION ----------------
    with st.spinner("Colorizing image... 🎨"):
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

    if mobile_view:
        st.subheader("🌈 Colorized Image")
        st.image(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖤 Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("🌈 Colorized")
            st.image(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ---------------- DOWNLOAD BUTTON ----------------
    _, buffer = cv2.imencode('.png', colorized)
    byte_io = io.BytesIO(buffer)

    st.download_button(
        label="⬇️ Download Colorized Image",
        data=byte_io,
        file_name="colorized.png",
        mime="image/png",
        use_container_width=True
    )

    st.success("✅ Colorization Complete!")