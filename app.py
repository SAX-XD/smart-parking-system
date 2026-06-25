import os
import tempfile

import streamlit as st
import cv2
import numpy as np

from src.pipeline import ParkingPipeline

st.set_page_config(page_title="Smart Parking Analytics", layout="wide")
st.title("🚗 Smart Parking Analytics System")

st.sidebar.title("🔧 System Settings")
st.sidebar.markdown("Configure your input stream and thresholds below.")


# --------------------------------------------------------------------------- #
# Pipeline (models load once and stay cached across reruns)
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

# --- The pipeline auto-tunes its own confidence thresholds per frame; the
# only thing worth exposing here is a speed/quality knob and a debug peek.
st.sidebar.subheader("🎯 Detection")
st.sidebar.caption("Confidence thresholds are picked automatically per image -- no tuning needed.")
max_width = st.sidebar.select_slider(
    "Processing width (smaller = faster, but can shift detection confidence)",
    options=[480, 640, 800, 960, 1280, "Original"],
    value="Original",
)
pipeline.configure(max_width=None if max_width == "Original" else max_width)

with st.sidebar.expander("Debug: thresholds picked for the last frame"):
    st.write(pipeline.last_thresholds)

input_mode = st.sidebar.selectbox(
    "Select Input Modality:",
    ("Static File Upload", "Live Webcam Feed"),
)


# --------------------------------------------------------------------------- #
# Dashboard metrics
# --------------------------------------------------------------------------- #

def display_dashboard_metrics(info):
    if not isinstance(info, dict):
        info = {"total_spaces": 0, "occupied_spaces": 0, "free_spaces": 0, "total_vehicles": 0}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Parking Slots", info.get("total_spaces", 0))
    col2.metric("Occupied Slots", info.get("occupied_spaces", 0))
    col3.metric("Available Slots", info.get("free_spaces", 0))
    col4.metric("Vehicles Detected", info.get("total_vehicles", 0))


# --------------------------------------------------------------------------- #
# Static file upload (image or video)
# --------------------------------------------------------------------------- #

if input_mode == "Static File Upload":
    st.subheader("📁 Static Image or Video Upload")
    uploaded_file = st.file_uploader(
        "Upload an image or video...",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        # Auto-reset the learned layout whenever a *different* file is uploaded,
        # so an old parking lot's bay layout never leaks into a new one.
        file_signature = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("last_uploaded_signature") != file_signature:
            pipeline.reset_memory()
            st.session_state["last_uploaded_signature"] = file_signature

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # --- Image ---------------------------------------------------------
        if suffix in (".jpg", ".jpeg", ".png", ".bmp"):
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Couldn't decode that image. Try a different file.")
            else:
                processed_img, info = pipeline.process_frame(image)
                display_dashboard_metrics(info)
                st.image(processed_img, channels="BGR", use_container_width=True)

        # --- Video -----------------------------------------------------------
        elif suffix in (".mp4", ".avi", ".mov", ".mkv"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file_bytes.tobytes())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    st.error("Could not open this video. The file may use a codec OpenCV can't decode.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
                    metric_placeholder = st.empty()
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0.0) if total_frames else None
                    stop_video = st.button("⏹ Stop")

                    frame_index = 0
                    try:
                        while cap.isOpened():
                            if stop_video:
                                break
                            ret, frame = cap.read()
                            if not ret:
                                break

                            processed_frame, info = pipeline.process_frame(frame)
                            with metric_placeholder.container():
                                display_dashboard_metrics(info)
                            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

                            frame_index += 1
                            if progress_bar and total_frames:
                                progress_bar.progress(min(frame_index / total_frames, 1.0))
                    finally:
                        cap.release()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            st.warning("Unsupported file type.")


# --------------------------------------------------------------------------- #
# Live webcam feed
# --------------------------------------------------------------------------- #

elif input_mode == "Live Webcam Feed":
    st.subheader("📹 Real-Time Video Stream")
    camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    run_live = st.checkbox("Start Live Stream", value=False)

    metric_placeholder = st.empty()
    video_placeholder = st.empty()

    if run_live:
        cap = cv2.VideoCapture(int(camera_index))
        try:
            if not cap.isOpened():
                st.error(f"Could not access webcam at index {camera_index}.")
            else:
                while cap.isOpened() and run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Webcam video stream interrupted.")
                        break

                    processed_frame, info = pipeline.process_frame(frame)
                    with metric_placeholder.container():
                        display_dashboard_metrics(info)
                    video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        finally:
            cap.release()
    else:
        st.info("Check the checkbox above to power on the webcam feed.")