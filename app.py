import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json
import time
from PIL import Image

# Paths
MODEL_PATH = "models/deepfake_image_model.h5"
HISTORY_FILE = "history.json"

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load or initialize history
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump({"history": []}, f)

def save_history(entry):
    with open(HISTORY_FILE, "r+") as f:
        data = json.load(f)
        data.setdefault("history", [])
        data["history"].insert(0, entry)
        data["history"] = data["history"][:200]  # keep latest 200
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f).get("history", [])

def predict(img: Image.Image):
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = float(model.predict(arr)[0][0])
    label = "Fake" if pred >= 0.5 else "Real"
    confidence = round(pred * 100, 2) if label == "Fake" else round((1 - pred) * 100, 2)
    return label, confidence, pred

# -------------------- UI --------------------

st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️", layout="wide")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    This app uses a trained deep learning model  
    to classify images as **Real** or **Fake**.
    
    **How to use:**
    1. Upload an image
    2. Click **Predict**
    3. See results and history
    """
)
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Model: `deepfake_image_model.h5`")
st.sidebar.write("📂 History stored in `history.json`")

# Main title
st.title("🕵️ DeepTruth - DeepFake Image Detector")
st.markdown("Upload an image to check if it's **Real or Fake**. Results will be saved in your history.")

# Upload section
uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Two column layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🔍 Prediction")
        if st.button("Run Prediction"):
            label, confidence, raw = predict(img)

            if label == "Fake":
                st.error(f"❌ Prediction: **{label}** ({confidence}% confidence)")
            else:
                st.success(f"✅ Prediction: **{label}** ({confidence}% confidence)")

            st.caption(f"Model raw output: `{raw:.4f}`")

            entry = {
                "filename": uploaded.name,
                "label": label,
                "confidence": confidence,
                "raw_value": raw,
                "timestamp": int(time.time()),
            }
            save_history(entry)

# Show history
st.markdown("---")
st.subheader("📜 Prediction History")

history = load_history()

if not history:
    st.info("No history yet. Upload an image and run prediction.")
else:
    import pandas as pd
    df = pd.DataFrame(history)
    df["time"] = df["timestamp"].apply(lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)))
    df = df[["filename", "label", "confidence", "time"]]
    st.dataframe(df, use_container_width=True, height=300)
