"""
streamlit_app.py
-----------------
VisionTalk demo: upload an image, get a generated caption underneath it.

Run:
    streamlit run app/streamlit_app.py

Requires a trained model at MODELS_DIR/visiontalk_model.keras and the
tokenizer bundle at MODELS_DIR/tokenizer.pkl (i.e. run download_dataset.py,
prepare_captions.py, extract_features.py, and train.py first).
"""

import os
import sys
import tempfile

import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import FINAL_MODEL_PATH, TOKENIZER_PATH  # noqa: E402

st.set_page_config(page_title="VisionTalk", page_icon="🖼️", layout="centered")
st.title("🖼️ VisionTalk — Image Captioning")
st.caption("CNN (VGG16) encoder + LSTM decoder, trained on Flickr8k.")


@st.cache_resource
def load_artifacts():
    from tensorflow.keras.models import load_model
    from inference import load_tokenizer_bundle

    if not os.path.isfile(FINAL_MODEL_PATH):
        return None, None
    model = load_model(FINAL_MODEL_PATH)
    tokenizer_bundle = load_tokenizer_bundle(TOKENIZER_PATH)
    return model, tokenizer_bundle


model, tokenizer_bundle = load_artifacts()

if model is None:
    st.error(
        "No trained model found yet.\n\n"
        "Run the training pipeline first:\n"
        "1. `python src/download_dataset.py`\n"
        "2. `python src/prepare_captions.py`\n"
        "3. `python src/extract_features.py`\n"
        "4. `python src/train.py`\n\n"
        f"Expected model at: `{FINAL_MODEL_PATH}`"
    )
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Generating caption..."):
        from inference import extract_single_image_feature, generate_caption

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            feature = extract_single_image_feature(tmp.name)
        os.unlink(tmp.name)

        caption = generate_caption(
            model, feature, tokenizer_bundle["tokenizer"], tokenizer_bundle["max_len"]
        )

    st.success(f"**Generated caption:** {caption}")
else:
    st.info("Upload a .jpg/.jpeg/.png image to get a caption.")
