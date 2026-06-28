"""
Core helper functions for Smart Parking Analytics.

CV Techniques implemented here:
  1. Image Enhancement  – enhance_frame()  (CLAHE on LAB lightness channel)
  2. Edge Detection     – apply_edge_overlay()  (Canny edges blended onto frame)
  3. Binary Morphology  – apply_morphology()  (dilation to clean edge maps)
  4. Object Detection   – draw_parking_bays()  (bounding boxes + polygon overlays)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence

# Modern BGR Color Palette
_FREE_COLOR     = (0, 210, 90)     # Vibrant Green
_OCCUPIED_COLOR = (0, 60, 220)     # Safety Red
_EDGE_COLOR     = (255, 220, 0)    # Cyan-Yellow for edge overlay


# ─── CV TECHNIQUE 1: Image Enhancement ────────────────────────────────────────
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on the L channel of LAB colour space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


# ─── CV TECHNIQUE 2 & 3: Edge Detection + Binary Morphology ───────────────────
def apply_edge_overlay(frame: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Canny edge detection with a morphological dilation cleanup pass,
    blended back onto the original frame as a semi-transparent overlay.

    Steps:
      1. Convert to greyscale
      2. Gaussian blur to reduce noise before edge detection
      3. Canny edge detection  ← CV Technique: Edge Detection
      4. Morphological dilation to connect nearby edge fragments  ← CV Technique: Binary Morphology
      5. Blend edge map onto colour frame at 30% opacity
    """
    grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(grey, (5, 5), 0)

    # Step 3 – Canny edge detection
    edges = cv2.Canny(blur, low_thresh, high_thresh)

    # Step 4 – Binary morphological dilation: thickens thin edges so they are
    # clearly visible on the output frame and connect broken edge segments
    kernel        = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Step 5 – Colour the edge map and blend onto the original frame
    edge_colour        = np.zeros_like(frame)
    edge_colour[edges_dilated > 0] = _EDGE_COLOR

    output = cv2.addWeighted(frame, 1.0, edge_colour, 0.30, 0)
    return output


# ─── Utility ──────────────────────────────────────────────────────────────────
def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


# ─── CV TECHNIQUE 4: Object Detection – parking bay annotation ────────────────
def draw_parking_bays(
    frame,
    polys,
    states,
    show_edges: bool = False,
    frame_counter: int = 0,
    fps_display: float = 0.0,
):
    """
    Annotate parking bays with coloured overlays.
    Optionally blends a Canny edge detection overlay onto the output.

    Parameters
    ----------
    frame         : BGR image to annotate
    polys         : list of polygon point lists for each parking space
    states        : list of bool – True = occupied, False = free
    show_edges    : if True, blend Canny edge overlay onto output
    frame_counter : current frame number shown in HUD
    fps_display   : current processing FPS shown in HUD
    """
    # Optionally apply edge overlay BEFORE drawing bays so bays sit on top
    base = apply_edge_overlay(frame) if show_edges else frame.copy()

    overlay = base.copy()
    output  = base.copy()

    for poly, occupied in zip(polys, states):
        color = _OCCUPIED_COLOR if occupied else _FREE_COLOR
        pts   = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(output, [pts], True, color, 2)

    # Blend the solid fills at 15% opacity
    cv2.addWeighted(overlay, 0.15, output, 0.85, 0, dst=output)

    # ── HUD Legend ────────────────────────────────────────────────────────
    hud_width = 360 if fps_display > 0 else 320
    cv2.rectangle(output, (15, 15), (hud_width, 55), (20, 20, 20), -1)

    cv2.circle(output, (35, 35), 8, _FREE_COLOR, -1)
    cv2.putText(output, "Available", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.circle(output, (175, 35), 8, _OCCUPIED_COLOR, -1)
    cv2.putText(output, "Occupied", (195, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Frame counter + FPS display (bottom-left) ─────────────────────────
    if frame_counter > 0 or fps_display > 0:
        h = output.shape[0]
        cv2.rectangle(output, (15, h - 35), (220, h - 10), (20, 20, 20), -1)
        hud_text = f"Frame: {frame_counter}"
        if fps_display > 0:
            hud_text += f"  |  FPS: {fps_display:.1f}"
        cv2.putText(
            output, hud_text,
            (22, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    return output


def extract_detections_by_class(result):
    detections = {}
    if result.boxes is not None:
        for i, cls in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            detections.setdefault(cls, []).append((poly, float(result.boxes.conf[i])))
    return detections
