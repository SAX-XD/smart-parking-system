from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np


Color = Tuple[int, int, int]
Point = Sequence[int]
Polygon = Sequence[Point]


def enhance_frame(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("enhance_frame received an empty frame.")

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("enhance_frame expects a BGR colour image with 3 channels.")

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced_l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def draw_parking_bays(
    frame: np.ndarray,
    space_coordinates: Sequence[Polygon],
    occupancy_states: Sequence[bool],
    alpha: float = 0.25,
    free_color: Color = (0, 255, 0),
    occupied_color: Color = (0, 0, 255),
    boundary_thickness: int = 2,
    font_scale: float = 0.55
) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("draw_parking_bays received an empty frame.")

    if len(space_coordinates) != len(occupancy_states):
        raise ValueError(
            "space_coordinates and occupancy_states must have the same length."
        )

    output = frame.copy()
    overlay = output.copy()

    polygon_arrays: List[np.ndarray] = [
        np.asarray(space, dtype=np.int32).reshape((-1, 1, 2))
        for space in space_coordinates
    ]

    for bay_index, polygon in enumerate(polygon_arrays):
        is_occupied = bool(occupancy_states[bay_index])
        color = occupied_color if is_occupied else free_color

        cv2.fillPoly(overlay, [polygon], color)

        cv2.polylines(
            output,
            [polygon],
            isClosed=True,
            color=color,
            thickness=boundary_thickness,
            lineType=cv2.LINE_AA
        )

        label_x = int(polygon[0][0][0])
        label_y = max(int(polygon[0][0][1]) - 10, 20)

        label_text = f"Bay {bay_index + 1}"
        cv2.putText(
            output,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    annotated_frame = cv2.addWeighted(
        overlay,
        alpha,
        output,
        1.0 - alpha,
        0
    )

    return annotated_frame


def count_available_slots(occupancy_states: Iterable[bool]) -> int:
    return sum(1 for occupied in occupancy_states if not occupied)
