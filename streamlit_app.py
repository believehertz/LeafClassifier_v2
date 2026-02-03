import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Leaf AI Pro", page_icon="🍃", layout="centered")

st.title("🍃 Leaf Classifier Pro")
st.caption("PyTorch + FastAPI + Streamlit")

API_URL = "http://localhost:8000"

# Check API
try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.json().get("model_loaded"):
        st.success("✅ API Ready")
    else:
        st.warning("⚠️ Model not loaded")
except:
    st.error("❌ Start API: python api.py")
    st.stop()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, use_container_width=True)
    
    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("AI thinking..."):
            files = {"file": (uploaded.name, uploaded, uploaded.type)}
            r = requests.post(f"{API_URL}/predict", files=files)
            
            if r.status_code == 200:
                result = r.json()
                pred = result["prediction"]
                conf = result["confidence"]
                
                with col2:
                    if pred == "LEAF":
                        st.success(f"### 🌿 {pred}!\n**{conf}%** confidence")
                        st.progress(int(conf))
                    else:
                        st.error(f"### 🚫 {pred}!\n**{conf}%** confidence")
                        st.progress(int(conf))
            else:
                st.error("Prediction failed")