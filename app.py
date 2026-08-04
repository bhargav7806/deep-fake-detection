import streamlit as st
import tensorflow as tf
from PIL import Image
from utils import process_image
import warnings

warnings.filterwarnings("ignore")

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fine_tuned_model.keras")

model = load_model()

THRESHOLD = 0.7

st.set_page_config(
    page_title="Deep Fake Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Deep Fake Detection")
st.write("Upload an image to check whether it is **Real** or **AI Generated**.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        processed = process_image(image)

        probability = model.predict(processed, verbose=0)[0][0]

        if probability >= THRESHOLD:
            st.success("✅ REAL IMAGE")
        else:
            st.error("❌ AI GENERATED IMAGE")

        st.write(f"**Probability:** {probability:.4f}")