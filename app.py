"""
Smart Parking Analytics – Streamlit Front-End with Targeted Interval Logic.
"""

import os
import tempfile
import cv2
import time
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
# Pipeline – cached so it only loads once across reruns
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_pipeline():
    return ParkingPipeline(
        vehicle_model_path="models/yolov8n.pt",
        space_model_path  ="models/best.pt",
    )

try:
    pipeline = load_pipeline()
except FileNotFoundError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Session state initialisation
# --------------------------------------------------------------------------- #
if "video_playing"       not in st.session_state:
    st.session_state.video_playing       = True
if "current_frame_index" not in st.session_state:
    st.session_state.current_frame_index = 0
if "paused_frame_image"  not in st.session_state:
    st.session_state.paused_frame_image  = None
if "paused_metrics_info" not in st.session_state:
    st.session_state.paused_metrics_info = {
        "total_spaces": 0, "occupied_spaces": 0,
        "free_spaces":  0, "total_vehicles":  0,
    }

# --------------------------------------------------------------------------- #
# Sidebar – settings
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")

space_conf = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Display Options")

enhance_contrast = st.sidebar.checkbox(
    "Enhance Contrast (Night Mode)",
    value=False,
    help="Applies CLAHE image enhancement to improve visibility in low-light conditions.",
)

show_edges = st.sidebar.checkbox(
    "Show Edge Detection Overlay",
    value=False,
    help="Applies Canny edge detection with morphological dilation and blends the result onto the frame.",
)

# Push all settings into the pipeline in one call
pipeline.configure(
    space_conf       = space_conf,
    enhance_contrast = enhance_contrast,
    show_edges       = show_edges,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input")
input_mode = st.sidebar.selectbox(
    "Source",
    ("Static File Upload", "Live Webcam Feed"),
)

# --------------------------------------------------------------------------- #
# Dashboard helper – now also shows detected vehicles
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

        # ── Image processing ───────────────────────────────────────────────
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            image_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image     = np.array(image_pil)
            image     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            with st.spinner("Processing…"):
                processed, info = pipeline.process_frame(image, run_inference=True, frame_counter=1)
            display_dashboard(info)
            st.image(processed, channels="BGR", use_container_width=True)

        # ── Video processing ───────────────────────────────────────────────
        elif suffix in (".mp4", ".avi", ".mov", ".mkv"):
            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes.tobytes())
                tmp_path = tmp.name

            # ── Playback control buttons ───────────────────────────────────
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
            play_btn    = ctrl_c1.button("▶ Play",    use_container_width=True)
            pause_btn   = ctrl_c2.button("⏸ Pause",  use_container_width=True)
            restart_btn = ctrl_c3.button("🔄 Restart", use_container_width=True)

            if play_btn:
                st.session_state.video_playing = True
            if pause_btn:
                st.session_state.video_playing = False
            if restart_btn:
                st.session_state.current_frame_index = 0
                st.session_state.paused_frame_image  = None
                st.session_state.video_playing       = True

            metric_ph = st.empty()
            video_ph  = st.empty()

            # ── Render paused state immediately without re-entering the loop
            if (
                not st.session_state.video_playing
                and st.session_state.paused_frame_image is not None
            ):
                with metric_ph.container():
                    display_dashboard(st.session_state.paused_metrics_info)
                video_ph.image(
                    st.session_state.paused_frame_image,
                    channels="BGR",
                    use_container_width=True,
                )

            cap = cv2.VideoCapture(tmp_path)

            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0 or np.isnan(native_fps):
                native_fps = 25.0

            total_frames            = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_delay             = 1.0 / native_fps
            inference_frame_interval = int(round(native_fps * 2.0))

            # Resume from wherever we paused
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_index)

            while cap.isOpened() and st.session_state.video_playing:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.session_state.current_frame_index = 0
                    st.session_state.paused_frame_image  = None
                    break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                st.session_state.current_frame_index = frame_idx

                is_interval   = ((frame_idx - 1) % inference_frame_interval == 0)
                is_last_frame = (frame_idx >= total_frames)
                should_run_ai = is_interval or is_last_frame

                processed, info = pipeline.process_frame(
                    frame,
                    run_inference = should_run_ai,
                    frame_counter = frame_idx,
                )

                # Always keep the latest frame in state for instant pause display
                st.session_state.paused_frame_image  = processed
                st.session_state.paused_metrics_info = info

                with metric_ph.container():
                    display_dashboard(info)
                video_ph.image(processed, channels="BGR", use_container_width=True)

                execution_time = time.time() - start_time
                delay_remainder = frame_delay - execution_time
                if delay_remainder > 0:
                    time.sleep(delay_remainder)

            cap.release()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# --------------------------------------------------------------------------- #
# Live webcam feed
# --------------------------------------------------------------------------- #

elif input_mode == "Live Webcam Feed":
    st.subheader("📹 Live Webcam Stream")
    camera_index = st.sidebar.number_input("Camera index", value=0, min_value=0, max_value=10, step=1)
    run_live     = st.checkbox("▶ Start Live Stream", value=False)

    metric_ph = st.empty()
    video_ph  = st.empty()

    if run_live:
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(f"⚠️ Could not open camera at index {int(camera_index)}. Try a different index.")
            st.stop()

        webcam_fps = cap.get(cv2.CAP_PROP_FPS)
        if webcam_fps <= 0 or np.isnan(webcam_fps):
            webcam_fps = 25.0

        inference_frame_interval = int(round(webcam_fps * 2.0))
        webcam_frame_count       = 0

        while cap.isOpened() and run_live:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame from webcam. Stream ended.")
                break

            webcam_frame_count += 1
            should_run_ai = ((webcam_frame_count - 1) % inference_frame_interval == 0)

            processed, info = pipeline.process_frame(
                frame,
                run_inference = should_run_ai,
                frame_counter = webcam_frame_count,
            )

            with metric_ph.container():
                display_dashboard(info)
            video_ph.image(processed, channels="BGR", use_container_width=True)

        cap.release()
