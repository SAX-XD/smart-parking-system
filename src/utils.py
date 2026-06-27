"""
Core helper functions for Smart Parking Analytics.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence

# Drawing settings
_FREE_COLOR     = (0, 210, 90)
_OCCUPIED_COLOR = (0, 60, 220)
_VEHICLE_COLOR  = (0, 165, 255)

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

def draw_parking_bays(frame, polys, states, vehicle_boxes=None):
    output = frame.copy()
    for poly, occupied in zip(polys, states):
        color = _OCCUPIED_COLOR if occupied else _FREE_COLOR
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(output, [pts], color)
        cv2.polylines(output, [pts], True, color, 2)
    
    if vehicle_boxes:
        for x1, y1, x2, y2 in vehicle_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), _VEHICLE_COLOR, 2)
            
    return cv2.addWeighted(output, 0.3, frame, 0.7, 0)

def extract_detections_by_class(result):
    detections = {}
    if result.boxes is not None:
        for i, cls in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            detections.setdefault(cls, []).append((poly, float(result.boxes.conf[i])))
    return detections