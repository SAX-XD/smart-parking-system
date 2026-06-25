"""
Shared computer-vision helpers for the Smart Parking Analytics pipeline.

Everything here is generic on purpose: there are no constants tied to a
specific image, video, or parking-lot layout. The same functions work for
a single static photo, a long video file, or a live webcam feed.
"""

from typing import Dict, Iterable, List, Sequence, Tuple
import cv2
import numpy as np

Color = Tuple[int, int, int]
Point = Sequence[float]
Polygon = Sequence[Point]
BoxXYXY = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Image pre-processing
# --------------------------------------------------------------------------- #

def enhance_frame(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the luminance channel only."""
    if frame is None or frame.size == 0:
        raise ValueError("enhance_frame received an empty frame.")
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("enhance_frame expects a BGR colour image with 3 channels.")

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a frame to max_width (keeping aspect ratio) for faster inference."""
    height, width = frame.shape[:2]
    if max_width is None or width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def polygon_to_xyxy(polygon: Polygon) -> BoxXYXY:
    """Bounding box (x1, y1, x2, y2) that encloses a polygon (or a box already)."""
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def box_area(box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: BoxXYXY, box_b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(box_a: BoxXYXY, box_b: BoxXYXY) -> float:
    inter = intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(slot_box: BoxXYXY, vehicle_box: BoxXYXY) -> float:
    """Fraction of the slot's own area that is covered by a vehicle box."""
    slot_area = box_area(slot_box)
    if slot_area <= 0:
        return 0.0
    return intersection_area(slot_box, vehicle_box) / slot_area


def deduplicate_boxes(box_lists: Sequence[Sequence[BoxXYXY]], iou_threshold: float = 0.5) -> List[BoxXYXY]:
    """
    Merge vehicle boxes coming from multiple detectors (e.g. the custom model
    and the generic YOLO model) into a single de-duplicated list, so the same
    physical car isn't counted twice.
    """
    all_boxes = [b for group in box_lists for b in group]
    all_boxes.sort(key=box_area, reverse=True)

    kept: List[BoxXYXY] = []
    for box in all_boxes:
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


# --------------------------------------------------------------------------- #
# YOLO result parsing
# --------------------------------------------------------------------------- #

def extract_detections_by_class(result) -> Dict[int, List[Tuple[Polygon, float]]]:
    """
    Turn a single Ultralytics `Results` object into {class_id: [(polygon, confidence), ...]}.

    Uses segmentation masks when available (real polygon outline). Falls back
    to the axis-aligned detection box when the model/result has no mask for
    that detection (e.g. a detection-only model, or a tiny/edge object).

    Confidence is returned per-detection (rather than filtered here) so the
    caller can apply a *different* threshold per class -- useful because two
    classes from the same model can have very different confidence
    distributions (e.g. a clearly-visible car vs. a partially-shadowed empty bay).
    """
    detections: Dict[int, List[Tuple[Polygon, float]]] = {}

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    mask_polys = result.masks.xy if result.masks is not None else None

    for i, cls_id in enumerate(cls_ids):
        if mask_polys is not None and i < len(mask_polys) and len(mask_polys[i]) > 2:
            polygon = mask_polys[i].astype(np.int32).tolist()
        else:
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        detections.setdefault(int(cls_id), []).append((polygon, float(confs[i])))

    return detections


def box_to_polygon(box: BoxXYXY) -> Polygon:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def auto_confidence_threshold(confidences: Sequence[float], min_floor: float = 0.05) -> float:
    """
    Pick a confidence cutoff *from the data itself* instead of a fixed guess.

    Real detections and background noise tend to form two separate clusters
    in confidence (e.g. real cars around 0.9+, stray noise around 0.05-0.1).
    Otsu's method -- normally used to split an image into foreground/background
    by intensity -- works just as well here: it finds the cutoff that best
    separates two clusters in *any* 1D set of numbers. We reuse OpenCV's
    implementation rather than re-deriving it.

    Falls back to `min_floor` when there isn't enough data to find a genuine
    split (zero or one detection, or all detections identical) -- in those
    cases there's nothing to separate, so we just apply a basic sanity floor.
    """
    if not confidences:
        return min_floor
    if len(confidences) == 1:
        return min_floor

    scaled = np.array([int(round(c * 255)) for c in confidences], dtype=np.uint8).reshape(-1, 1)
    if scaled.max() == scaled.min():
        return min_floor

    otsu_value, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(otsu_value / 255.0, min_floor)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_parking_bays(
    frame: np.ndarray,
    space_coordinates: Sequence[Polygon],
    occupancy_states: Sequence[bool],
    alpha: float = 0.30,
    free_color: Color = (0, 200, 0),
    occupied_color: Color = (0, 0, 230),
    boundary_thickness: int = 2,
    font_scale: float = 0.55,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw filled + outlined parking bays, colour-coded by occupancy."""
    if frame is None or frame.size == 0:
        raise ValueError("draw_parking_bays received an empty frame.")
    if len(space_coordinates) != len(occupancy_states):
        raise ValueError("space_coordinates and occupancy_states must have the same length.")

    output = frame.copy()
    overlay = output.copy()

    polygon_arrays: List[np.ndarray] = [
        np.array(space, dtype=np.int32).reshape((-1, 1, 2))
        for space in space_coordinates
    ]

    for bay_index, polygon in enumerate(polygon_arrays):
        is_occupied = bool(occupancy_states[bay_index])
        color = occupied_color if is_occupied else free_color

        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(output, [polygon], isClosed=True, color=color,
                      thickness=boundary_thickness, lineType=cv2.LINE_AA)

        label_x = int(polygon[0][0][0])
        label_y = max(int(polygon[0][0][1]) - 10, 20)
        cv2.putText(output, f"Bay {bay_index + 1}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness=2, lineType=cv2.LINE_AA)

    annotated_frame = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    if show_legend:
        draw_legend(annotated_frame, free_color, occupied_color)

    return annotated_frame


def draw_legend(frame: np.ndarray, free_color: Color, occupied_color: Color) -> None:
    """Small in-frame legend, sized relative to the frame so it never dominates it."""
    h, w = frame.shape[:2]
    box_w = int(np.clip(w * 0.09, 80, 150))
    box_h = int(box_w * 0.42)
    margin = max(6, int(box_w * 0.06))
    swatch = max(8, int(box_h * 0.28))
    font_scale = max(0.32, box_h / 130)
    gap = box_h // 2

    x0, y0 = margin, margin
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)

    sx, sy = x0 + swatch // 2, y0 + gap // 2 + swatch // 4
    cv2.rectangle(frame, (sx, sy), (sx + swatch, sy + swatch), free_color, -1)
    cv2.putText(frame, "Free", (sx + swatch + 6, sy + swatch - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    sy2 = sy + gap
    cv2.rectangle(frame, (sx, sy2), (sx + swatch, sy2 + swatch), occupied_color, -1)
    cv2.putText(frame, "Occupied", (sx + swatch + 6, sy2 + swatch - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def count_available_slots(occupancy_states: Iterable[bool]) -> int:
    return sum(1 for occupied in occupancy_states if not occupied)