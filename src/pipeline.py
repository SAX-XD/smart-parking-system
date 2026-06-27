"""
Smart Parking Analytics – two-model pipeline.
"""

import os
import numpy as np
from ultralytics import YOLO
from utils import (
    enhance_frame,
    resize_keep_aspect,
    draw_parking_bays,
    extract_detections_by_class,
    polygon_to_xyxy,
    box_to_polygon,
    deduplicate_boxes,
)

# Configuration for best.pt
SPACE_OCCUPIED_ID = 1
SPACE_EMPTY_ID    = 0
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

class ParkingPipeline:
    def __init__(self, vehicle_model_path="models/yolov8n.pt", space_model_path="models/best.pt"):
        if not os.path.exists(vehicle_model_path) or not os.path.exists(space_model_path):
            raise FileNotFoundError("Model files not found in models/ directory.")

        self.space_model   = YOLO(space_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        
        self.max_width = None
        self.enhance_contrast = False
        self.space_conf = 0.25
        self.vehicle_conf = 0.25
        self.show_vehicle_boxes = False

    def configure(self, max_width=None, enhance_contrast=None, space_conf=None, vehicle_conf=None, show_vehicle_boxes=None):
        if max_width is not None: self.max_width = max_width
        if enhance_contrast is not None: self.enhance_contrast = enhance_contrast
        if space_conf is not None: self.space_conf = space_conf
        if vehicle_conf is not None: self.vehicle_conf = vehicle_conf
        if show_vehicle_boxes is not None: self.show_vehicle_boxes = show_vehicle_boxes

    def process_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return frame, {"total_spaces": 0, "occupied_spaces": 0, "free_spaces": 0, "total_vehicles": 0}

        if self.max_width:
            frame = resize_keep_aspect(frame, self.max_width)
        work = enhance_frame(frame) if self.enhance_contrast else frame

        # Detections
        space_results = self.space_model(work, conf=self.space_conf, verbose=False)[0]
        by_class = extract_detections_by_class(space_results)
        
        vehicle_results = self.vehicle_model(work, classes=COCO_VEHICLE_CLASSES, conf=self.vehicle_conf, verbose=False)[0]
        coco_boxes = [tuple(map(int, b)) for b in vehicle_results.boxes.xyxy.cpu().numpy()] if vehicle_results.boxes is not None else []

        # Process as whole_bay
        occ_polys = [poly for poly, _ in by_class.get(SPACE_OCCUPIED_ID, [])]
        emp_polys = [poly for poly, _ in by_class.get(SPACE_EMPTY_ID, [])]

        total = len(occ_polys) + len(emp_polys)
        states = [True] * len(occ_polys) + [False] * len(emp_polys)
        
        annotated = draw_parking_bays(frame, occ_polys + emp_polys, states, 
                                     vehicle_boxes=coco_boxes if self.show_vehicle_boxes else None)

        return annotated, {
            "total_spaces": total,
            "occupied_spaces": len(occ_polys),
            "free_spaces": len(emp_polys),
            "total_vehicles": len(coco_boxes)
        }