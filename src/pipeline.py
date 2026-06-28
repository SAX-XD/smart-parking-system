"""
Smart Parking Analytics – Two-Model Pipeline with Selective Interval Inference.

CV Techniques orchestrated here:
  1. Image Enhancement    – CLAHE via enhance_frame()
  2. Edge Detection       – Canny via apply_edge_overlay() (togglable)
  3. Binary Morphology    – Dilation inside apply_edge_overlay()
  4. Object Detection     – YOLOv8 space classifier + COCO vehicle detector
"""

import os
import time
import numpy as np
from ultralytics import YOLO
from .utils import (
    enhance_frame,
    resize_keep_aspect,
    draw_parking_bays,
    extract_detections_by_class,
)

# Class IDs from custom best.pt weights
SPACE_OCCUPIED_ID    = 0
SPACE_EMPTY_ID       = 1
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


class ParkingPipeline:
    def __init__(
        self,
        vehicle_model_path: str = "models/yolov8n.pt",
        space_model_path:   str = "models/best.pt",
    ):
        if not os.path.exists(vehicle_model_path) or not os.path.exists(space_model_path):
            raise FileNotFoundError("Model files not found in models/ directory.")

        self.space_model   = YOLO(space_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)

        # ── Configurable flags ─────────────────────────────────────────────
        self.max_width        = None
        self.enhance_contrast = False
        self.space_conf       = 0.25
        self.vehicle_conf     = 0.25
        self.show_edges       = False   # toggle Canny edge overlay

        # ── Caching: avoid dropping display between inference intervals ────
        self.cached_metrics  = {
            "total_spaces": 0, "occupied_spaces": 0,
            "free_spaces": 0,  "total_vehicles":  0,
        }
        self.cached_occ_polys = []
        self.cached_emp_polys = []
        self.cached_coco_boxes = []

        # ── FPS tracking ──────────────────────────────────────────────────
        self._fps_timestamps: list = []   # rolling window of recent frame times

    # ── Public configuration method ───────────────────────────────────────
    def configure(
        self,
        max_width        = None,
        enhance_contrast = None,
        space_conf       = None,
        vehicle_conf     = None,
        show_edges       = None,
    ):
        if max_width         is not None: self.max_width        = max_width
        if enhance_contrast  is not None: self.enhance_contrast = enhance_contrast
        if space_conf        is not None: self.space_conf       = space_conf
        if vehicle_conf      is not None: self.vehicle_conf     = vehicle_conf
        if show_edges        is not None: self.show_edges       = show_edges

    # ── FPS helper ────────────────────────────────────────────────────────
    def _update_fps(self) -> float:
        """Return rolling average FPS over the last 30 frames."""
        now = time.time()
        self._fps_timestamps.append(now)
        # Keep only the last 30 timestamps
        if len(self._fps_timestamps) > 30:
            self._fps_timestamps = self._fps_timestamps[-30:]
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        return (len(self._fps_timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    # ── Main processing entry point ───────────────────────────────────────
    def process_frame(
        self,
        frame:         np.ndarray,
        run_inference: bool = True,
        frame_counter: int  = 0,
    ):
        """
        Process a single BGR frame.

        Parameters
        ----------
        frame         : raw BGR image from cv2 / PIL
        run_inference : False on non-interval frames – skip model, use cache
        frame_counter : frame number to display in HUD

        Returns
        -------
        annotated_frame : BGR image with all overlays drawn
        metrics         : dict with total/occupied/free/vehicle counts
        """
        if frame is None or frame.size == 0:
            empty = {"total_spaces": 0, "occupied_spaces": 0, "free_spaces": 0, "total_vehicles": 0}
            return frame, empty

        if self.max_width:
            frame = resize_keep_aspect(frame, self.max_width)

        fps = self._update_fps()

        # ── Cache path: skip model, redraw cached polygons on new frame ───
        if not run_inference and self.cached_metrics["total_spaces"] > 0:
            annotated = draw_parking_bays(
                frame,
                self.cached_occ_polys + self.cached_emp_polys,
                [True]  * len(self.cached_occ_polys) +
                [False] * len(self.cached_emp_polys),
                show_edges    = self.show_edges,
                frame_counter = frame_counter,
                fps_display   = fps,
            )
            return annotated, self.cached_metrics

        # ── Inference path ────────────────────────────────────────────────
        work = enhance_frame(frame) if self.enhance_contrast else frame

        # Parking-space classification model (custom fine-tuned)
        space_results = self.space_model(work, conf=self.space_conf, verbose=False)[0]
        by_class      = extract_detections_by_class(space_results)

        # General vehicle detection model (COCO pre-trained YOLOv8n)
        vehicle_results = self.vehicle_model(
            work,
            classes = COCO_VEHICLE_CLASSES,
            conf    = self.vehicle_conf,
            verbose = False,
        )[0]
        coco_boxes = (
            [tuple(map(int, b)) for b in vehicle_results.boxes.xyxy.cpu().numpy()]
            if vehicle_results.boxes is not None else []
        )

        occ_polys = [poly for poly, _ in by_class.get(SPACE_OCCUPIED_ID, [])]
        emp_polys = [poly for poly, _ in by_class.get(SPACE_EMPTY_ID,    [])]

        total  = len(occ_polys) + len(emp_polys)
        states = [True] * len(occ_polys) + [False] * len(emp_polys)

        annotated = draw_parking_bays(
            frame,
            occ_polys + emp_polys,
            states,
            show_edges    = self.show_edges,
            frame_counter = frame_counter,
            fps_display   = fps,
        )

        metrics = {
            "total_spaces":    total,
            "occupied_spaces": len(occ_polys),
            "free_spaces":     len(emp_polys),
            "total_vehicles":  len(coco_boxes),
        }

        # Update cache for non-inference frames
        self.cached_metrics   = metrics
        self.cached_occ_polys = occ_polys
        self.cached_emp_polys = emp_polys
        self.cached_coco_boxes= coco_boxes

        return annotated, metrics
