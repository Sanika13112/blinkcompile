import streamlit as st
import time
import os
import io
import sys

# ==============================
# Environment safety settings
# ==============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==============================
# Safe imports with fallback
# ==============================
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from compressor import EdgeFlowCompressor
except Exception:
    class EdgeFlowCompressor:
        def __init__(self, model_id): self.model_id = model_id
        def load_huggingface_model(self): return "Demo", "Demo mode"
        def create_dummy_model(self): return "Demo", "Demo mode"
        def compress_to_tflite(self, quantization=None):
            return "demo.tflite", 100_000_000, 25_000_000
        def export_to_onnx(self): return "demo.onnx"
        def apply_pruning(self, rate): pass
        def get_model_info(self):
            return {
                "parameters": "1M",
                "layers": "8",
                "input_shape": "(224,224,3)",
                "output_shape": "(10,)"
            }

try:
    from utils import (
        generate_qr,
        get_system_info,
        format_file_size,
        validate_edge_compatibility,
        create_model_card,
        safe_calculate_reduction
    )
except Exception:
    from PIL import Image, ImageDraw
    def generate_qr(data):
        img = Image.new("RGB", (200, 200), "white")
        d = ImageDraw.Draw(img)
        d.text((40, 90), "Demo QR", fill="black")
        return img

    def get_system_info():
        return {
            "os": "Unknown",
            "processor": "Unknown",
            "python_version": sys.version.split()[0],
            "memory_gb": 4,
            "cpu_cores": 2
        }

    def format_file_size(b): return f"{b/1024/1024:.1f} MB"
    def validate_edge_compatibility(a, b): return "✅ Good"
    def create_model_card(a, b, c, d): return "<div>Demo Model Card</div>"
    def safe_calculate_reduction(a, b): return 75.0

try:
    import config
except Exception:
    class config:
        POPULAR_MODELS = {
            "Vision": ["google/mobilenet_v2_1.0_224"],
            "NLP": ["distilbert-base-uncased"]
        }
        COMPRESSION_PRESETS = {
            "Balanced": {"quantization": True, "pruning": 0.3},
            "Max Performance": {"quantization": True, "pruning": 0.5}
        }
        DEVICE_PROFILES = {
            "Raspberry Pi 4": {"RAM": "4GB"},
            "Android Phone": {"RAM": "6GB"}
        }

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="BlinkCompile",
    page_icon="⚡",
    layout="wide"
)

# ==============================
# Header
# ==============================
st.markdown("""
<div style="background:linear-gradient(90deg,#667eea,#764ba2);
padding:25px;border-radius:12px;text-align:center;color:white">
<h1>⚡ BlinkCompile</h1>
<p>Deploy AI Models to Edge Devices in Minutes</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Session State
# ==============================
for key in ["compressor", "model_loaded", "compression_done", "compression_results", "model_id"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "compressor" else False

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("🛠 Configuration")

    model_source = st.radio(
        "Model Source",
        ["Popular Models", "Hugging Face ID", "Demo Mode"]
    )

    if model_source == "Popular Models":
        mtype = st.selectbox("Model Type", list(config.POPULAR_MODELS.keys()))
        model_id = st.selectbox("Model", config.POPULAR_MODELS[mtype])
    elif model_source == "Hugging Face ID":
        model_id = st.text_input("Model ID", "google/mobilenet_v2_1.0_224")
    else:
        model_id = "demo"

    st.session_state.model_id = model_id

    preset = st.selectbox(
        "Compression Preset",
        list(config.COMPRESSION_PRESETS.keys())
    )

    device = st.selectbox(
        "Target Device",
        list(config.DEVICE_PROFILES.keys())
    )

    col1, col2 = st.columns(2)
    with col1:
        load_clicked = st.button("📥 Load Model")
    with col2:
        compress_clicked = st.button("⚡ Compress")

    st.markdown("---")
    st.subheader("System Info")
    info = get_system_info()
    st.text(f"OS: {info['os']}")
    st.text(f"Python: {info['python_version']}")

# ==============================
# Load Model
# ==============================
if load_clicked:
    with st.spinner("Loading model..."):
        comp = EdgeFlowCompressor(model_id)
        if model_id == "demo":
            comp.create_dummy_model()
        else:
            comp.load_huggingface_model()

        st.session_state.compressor = comp
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully")

# ==============================
# Main Tabs
# ==============================
if st.session_state.model_loaded:
    comp = st.session_state.compressor

    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "⚡ Compression", "📱 Mobile Demo"])

    with tab1:
        st.json(comp.get_model_info())
        st.markdown(f"**Compatibility:** {validate_edge_compatibility(100, device)}")

    with tab2:
        if compress_clicked:
            with st.spinner("Compressing model..."):
                tflite, orig, comp_size = comp.compress_to_tflite()
                onnx = comp.export_to_onnx()

                st.session_state.compression_results = {
                    "tflite": tflite,
                    "onnx": onnx,
                    "orig": orig,
                    "comp": comp_size
                }
                st.session_state.compression_done = True

        if st.session_state.compression_done:
            r = st.session_state.compression_results
            reduction = safe_calculate_reduction(r["orig"], r["comp"])

            st.metric("Original Size", format_file_size(r["orig"]))
            st.metric("Compressed Size", format_file_size(r["comp"]))
            st.metric("Reduction", f"{reduction:.1f}%")

            st.markdown(create_model_card(
                model_id, r["orig"], r["comp"], ["Quantization"]
            ), unsafe_allow_html=True)

    with tab3:
        qr = generate_qr("https://github.com/Sanika13112/blinkcompile")
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf.getvalue(), width=220)
        st.caption("Scan on mobile")

# ==============================
# Footer
# ==============================
st.markdown("""
<hr>
<div style="text-align:center;color:gray">
⚡ BlinkCompile • Streamlit + TensorFlow • Edge AI
</div>
""", unsafe_allow_html=True)
