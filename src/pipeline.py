import os
import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    enhance_frame,
    resize_keep_aspect,
    draw_parking_bays,
    extract_detections_by_class,
    auto_confidence_threshold,
    polygon_to_xyxy,
    box_to_polygon,
    deduplicate_boxes,
)

# Class indices inside the custom-trained segmentation model (models/best.pt),
# confirmed from the checkpoint's own metadata: {0: 'car', 1: 'parking-space'}.
#
# Tested against real sample footage: the 'parking-space' class only ever
# fires on bays with NO car on them (visibly empty ground/markings). There is
# no separate "occupied bay" outline in this dataset -- a car detection IS
# the occupied signal. So:
#     occupied_spaces = number of vehicles detected (deduplicated across both models)
#     free_spaces     = number of 'parking-space' detections from the custom model
#     total_spaces    = occupied_spaces + free_spaces
CAR_CLASS = 0
EMPTY_SPACE_CLASS = 1

# COCO class indices used from the generic yolov8n.pt vehicle detector.
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Very low floor passed to the models themselves, just to pull in every
# candidate detection (real + noise) so the auto-thresholding below has
# enough data points to find a genuine split between the two.
DETECTION_FLOOR_CONF = 0.01

# Safety floors for the auto-picked threshold -- not "the" threshold, just a
# backstop for when there's too little data to find a real split (e.g. a
# single detection) so obvious noise still can't sneak through.
VEHICLE_MIN_FLOOR = 0.25
EMPTY_SPACE_MIN_FLOOR = 0.05


class ParkingPipeline:
    """
    Two-model parking-occupancy pipeline with self-tuning confidence.

    Model 1 (custom, best.pt): segmentation model trained on this parking lot.
        - class 'car'           -> a parked vehicle (occupies one bay)
        - class 'parking-space' -> a bay with nothing parked on it
    Model 2 (generic, yolov8n.pt): pretrained COCO detector, used as a second,
        independent opinion on where the vehicles are.

    Rather than a fixed confidence cutoff, every class's detections are
    pulled in down to a near-zero floor, then split into "real" vs "noise"
    using Otsu's method on the confidence values themselves (see
    `utils.auto_confidence_threshold`). This adapts per image/frame instead
    of needing a slider tuned per parking lot or lighting condition.
    """

    def __init__(
        self,
        vehicle_model_path: str = "models/yolov8n.pt",
        space_model_path: str = "models/best.pt",
        max_width: int = None,
        enhance_contrast: bool = False,
    ):
        for path, label in ((vehicle_model_path, "vehicle"), (space_model_path, "parking-space")):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Could not find the {label} model at '{path}'. "
                    f"Check that the file exists relative to your working directory."
                )

        self.vehicle_model = YOLO(vehicle_model_path)
        self.space_model = YOLO(space_model_path)

        self.max_width = max_width
        # Off by default: testing against real footage showed CLAHE contrast
        # enhancement (designed for poor-lighting webcam input) actually
        # *lowers* the custom model's confidence on well-lit frames, since it
        # was trained on raw, unenhanced images. Turn on only for genuinely
        # dark/low-contrast footage.
        self.enhance_contrast = enhance_contrast

        # Last thresholds the algorithm picked, exposed purely for debugging/
        # display -- nothing reads these back into detection logic.
        self.last_thresholds = {"vehicle": None, "empty_space": None}

    # ------------------------------------------------------------------ #
    # Public controls (wired up to the Streamlit sidebar)
    # ------------------------------------------------------------------ #

    def configure(self, max_width=None, enhance_contrast=None):
        """Cheap to call every rerun -- only updates plain attributes, never reloads models."""
        if max_width is not None:
            self.max_width = max_width
        if enhance_contrast is not None:
            self.enhance_contrast = enhance_contrast

    def reset_memory(self):
        """
        Kept for backward compatibility with the UI's Reset button. The
        pipeline no longer carries any state between frames -- every frame
        is analysed independently -- so there's nothing to actually clear.
        """
        pass

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return frame, {"total_spaces": 0, "occupied_spaces": 0, "free_spaces": 0, "total_vehicles": 0}

        if self.max_width:
            frame = resize_keep_aspect(frame, self.max_width)

        enhanced_frame = enhance_frame(frame) if self.enhance_contrast else frame

        # --- Custom model: cast a wide net, then auto-split each class -----
        space_results = self.space_model(enhanced_frame, conf=DETECTION_FLOOR_CONF, verbose=False)[0]
        by_class = extract_detections_by_class(space_results)

        car_candidates = by_class.get(CAR_CLASS, [])
        empty_candidates = by_class.get(EMPTY_SPACE_CLASS, [])

        car_cutoff = auto_confidence_threshold([c for _, c in car_candidates], min_floor=VEHICLE_MIN_FLOOR)
        empty_cutoff = auto_confidence_threshold([c for _, c in empty_candidates], min_floor=EMPTY_SPACE_MIN_FLOOR)

        custom_car_polygons = [poly for poly, conf in car_candidates if conf >= car_cutoff]
        empty_space_polygons = [poly for poly, conf in empty_candidates if conf >= empty_cutoff]

        # --- Generic COCO model: same wide-net-then-split treatment --------
        vehicle_results = self.vehicle_model(
            enhanced_frame, classes=COCO_VEHICLE_CLASSES, conf=DETECTION_FLOOR_CONF, verbose=False
        )[0]
        coco_boxes, coco_confs = [], []
        if vehicle_results.boxes is not None and len(vehicle_results.boxes) > 0:
            coco_boxes = [tuple(map(int, b)) for b in vehicle_results.boxes.xyxy.cpu().numpy()]
            coco_confs = vehicle_results.boxes.conf.cpu().numpy().tolist()

        coco_cutoff = auto_confidence_threshold(coco_confs, min_floor=VEHICLE_MIN_FLOOR)
        coco_vehicle_boxes = [b for b, c in zip(coco_boxes, coco_confs) if c >= coco_cutoff]

        self.last_thresholds = {"vehicle": round(max(car_cutoff, coco_cutoff), 3), "empty_space": round(empty_cutoff, 3)}

        # --- Combine + de-duplicate vehicles seen by either model ----------
        custom_car_boxes = [polygon_to_xyxy(p) for p in custom_car_polygons]
        vehicle_boxes = deduplicate_boxes([custom_car_boxes, coco_vehicle_boxes], iou_threshold=0.5)

        occupied_spaces = len(vehicle_boxes)
        free_spaces = len(empty_space_polygons)
        total_spaces = occupied_spaces + free_spaces

        # Draw both signals on one overlay: vehicles -> "occupied" (red),
        # detected empty ground -> "free" (green).
        all_polygons = [box_to_polygon(b) for b in vehicle_boxes] + list(empty_space_polygons)
        occupancy_states = [True] * occupied_spaces + [False] * free_spaces

        annotated_frame = (
            draw_parking_bays(frame, all_polygons, occupancy_states)
            if total_spaces > 0 else frame.copy()
        )

        info = {
            "total_spaces": total_spaces,
            "occupied_spaces": occupied_spaces,
            "free_spaces": free_spaces,
            "total_vehicles": occupied_spaces,
        }
        return annotated_frame, info