"""
Smart Parking Analytics – Streamlit front-end.
"""

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

from src.pipeline import ParkingPipeline

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 Smart Parking Analytics")

# --------------------------------------------------------------------------- #
# Pipeline – cached
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_pipeline():
    return ParkingPipeline(
        vehicle_model_path="models/yolov8n.pt",
        space_model_path="models/best.pt",
    )

try:
    pipeline = load_pipeline()
except FileNotFoundError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar – settings
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")

space_conf = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
)

enhance_contrast = st.sidebar.checkbox(
    "Enhance Contrast (Night Mode)",
    value=False,
)

pipeline.configure(
    space_conf=space_conf,
    enhance_contrast=enhance_contrast,
)

st.sidebar.subheader("📥 Input")
input_mode = st.sidebar.selectbox(
    "Source",
    ("Static File Upload", "Live Webcam Feed"),
)

# --------------------------------------------------------------------------- #
# Dashboard helper
# --------------------------------------------------------------------------- #

def display_dashboard(info: dict):
    total    = info.get("total_spaces",    0)
    occupied = info.get("occupied_spaces", 0)
    free     = info.get("free_spaces",     0)

    c1, c2, c3 = st.columns(3)
    c1.metric("🅿️ Total Bays", total)
    c2.metric("🔴 Occupied",   occupied)
    c3.metric("🟢 Available",  free)

    if total > 0:
        pct = int(occupied / total * 100)
        st.progress(pct, text=f"Lot occupancy: {pct}%")

# --------------------------------------------------------------------------- #
# Static file upload
# --------------------------------------------------------------------------- #

if input_mode == "Static File Upload":
    st.subheader("📁 Upload Image or Video")
    uploaded_file = st.file_uploader(
        "Drag and drop a file",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        suffix     = os.path.splitext(uploaded_file.name)[1].lower()

        # ── Image (Unified processing for JPG, PNG, BMP, WEBP) ──────────────
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            image_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image = np.array(image_pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            with st.spinner("Processing…"):
                processed, info = pipeline.process_frame(image)
            display_dashboard(info)
            st.image(processed, channels="BGR", use_container_width=True)

        # ── Video ───────────────────────────────────────────────────────────
        elif suffix in (".mp4", ".avi", ".mov", ".mkv"):
            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes.tobytes())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            metric_ph  = st.empty()
            video_ph   = st.empty()
            stop_btn   = st.button("⏹ Stop video")

            while cap.isOpened():
                if stop_btn: break
                ret, frame = cap.read()
                if not ret: break
                processed, info = pipeline.process_frame(frame)
                with metric_ph.container():
                    display_dashboard(info)
                video_ph.image(processed, channels="BGR", use_container_width=True)
            cap.release()
            os.remove(tmp_path)

# --------------------------------------------------------------------------- #
# Live webcam feed
# --------------------------------------------------------------------------- #

elif input_mode == "Live Webcam Feed":
    st.subheader("📹 Live Webcam Stream")
    camera_index = st.sidebar.number_input("Camera index", value=0)
    run_live = st.checkbox("▶ Start Live Stream", value=False)

    metric_ph = st.empty()
    video_ph  = st.empty()

    if run_live:
        cap = cv2.VideoCapture(int(camera_index))
        while cap.isOpened() and run_live:
            ret, frame = cap.read()
            if not ret: break
            processed, info = pipeline.process_frame(frame)
            with metric_ph.container():
                display_dashboard(info)
            video_ph.image(processed, channels="BGR", use_container_width=True)
        cap.release()