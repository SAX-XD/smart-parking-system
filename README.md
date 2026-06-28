# 🚗 Smart Parking Analytics System

An advanced, dual-model computer vision pipeline and real-time analytical dashboard designed to monitor parking lot occupancy. This system seamlessly orchestrates traditional digital image processing operations alongside modern deep learning architectures to provide ultra-fast, optimized tracking over live video streams, static uploads, and webcams.

---

## 🛠️ Integrated Computer Vision Capabilities

This project implements a comprehensive suite of foundational and modern Computer Vision (CV) techniques:

* **Image Enhancement (CLAHE):** Converts input frames to the LAB color space and applies Contrast Limited Adaptive Histogram Equalization on the lightness (L) channel to normalize poor, variable, or low-light night-time conditions.

* **Edge Detection (Canny):** Pinpoints fine physical and structural boundaries (e.g., painted parking bay dividers) by computing localized pixel intensity spatial gradients.

* **Binary Morphological Operations:** Performs morphological dilation with a structured rectangular kernel over binary edge maps to bridge fractured pixel gaps and thicken faint contours for high-definition rendering.

* **Object Detection (Deep Learning):** Leverages a specialized, fine-tuned YOLOv8 classification engine to accurately isolate and determine state spaces (`SPACE_OCCUPIED` vs `SPACE_EMPTY`).

* **Video Processing & Dynamic Inference:** Decodes sequential temporal video arrays while orchestrating a highly optimized frame-skipping interval strategy (calculating AI logic strictly every 2 seconds) to boost throughput efficiency.

* **State Tracking (Caching Architecture):** Utilizes a low-latency structural state-caching matrix to maintain and carry forward spatial metrics on non-inference frames, completely eliminating rendering stutter.

---

## 📁 Repository Structure

```text
smart-parking-system/
├── src/
│   ├── __init__.py
│   ├── pipeline.py       # Core orchestration class & caching engine
│   └── utils.py          # Classical CV algorithms (CLAHE, Canny, Dilation)
├── test_images_videos/   # Packaged sample media clips for testing
├── training/             # Model weights (best.pt and yolov8n.pt)
├── .gitignore            # Excludes local virtual environment (.venv)
├── app.py                # Streamlit responsive front-end dashboard
├── requirements.txt      # Multi-platform library dependencies
└── run_app.bat           # One-click Windows deployment script
🚀 Quick Start & Deployment Instructions
Method A: One-Click Windows Execution (Recommended)
If you are on a Windows machine and have already initialized your environment, simply double-click the automation script located in the root folder:

run_app.bat

Method B: Manual Terminal Execution
If executing across Linux, macOS, or clean environments, follow these structured steps:

Navigate to the Project Root:

Bash
cd smart-parking-system
Initialize a Virtual Environment:

Bash
python -m venv .venv
Activate the Environment:

Windows: .venv\Scripts\activate

macOS/Linux: source .venv/bin/activate

Install Required Framework Dependencies:

Bash
pip install -r requirements.txt
Boot up the Streamlit Server Application:

Bash
streamlit run app.py
🎛️ Live Dashboard Operations Guide
Source Selector: Swap instantly between handling high-resolution static media files and active live hardware webcam feeds.

Detection Sensitivity Slider: Adjusts confidence thresholds dynamically for custom-space classifications.

Enhance Contrast Checkbox: Toggles the localized CLAHE matrix to clarify dark or underexposed feeds.

Show Edge Detection Overlay Checkbox: Toggles live Canny edge maps and morphological dilations blended seamlessly onto the output feed at 30% opacity.

Interactive Controls: Use the responsive Play, Pause, and Restart buttons to evaluate deep metric properties frame-by-frame.
