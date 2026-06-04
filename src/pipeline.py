import cv2
import numpy as np
from ultralytics import YOLO

class ParkingPipeline:
    def __init__(self, custom_model_path=None):
        # Model 1: Pre-trained out-of-the-box vehicle detector
        self.vehicle_model = YOLO('yolov8n.pt') 
        
        # Model 2: Member 3's custom model (using base model as a fallback placeholder)
        if custom_model_path:
            self.custom_space_model = YOLO(custom_model_path)
        else:
            self.custom_space_model = None

    def process_frame(self, frame):
        """
        Main entry point that Member 1's UI will call inside the processing loop.
        Takes a raw OpenCV BGR frame, processes it, and returns (annotated_frame, available_slots)
        """
        # 1. Image Enhancement Task (Placeholder for Member 2's CLAHE code)
        enhanced_frame = frame.copy() 
        
        # 2. Vehicle Detection (Model 1)
        # Run inference on classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        vehicle_results = self.vehicle_model(enhanced_frame, classes=[2, 3, 5, 7], verbose=False)
        
        # 3. Space Tracking Logic
        # Temporary mock coordinates for 3 parking spaces until Member 3 finishes training
        # Format: List of 4-point polygons [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        mock_spaces = [
            [[50, 300], [200, 300], [200, 450], [50, 450]],
            [[250, 300], [400, 300], [400, 450], [250, 450]],
            [[450, 300], [600, 300], [600, 450], [450, 450]]
        ]
        # Temporary placeholder occupancy states: False = empty, True = occupied
        occupancy_states = [False, False, False] 
        
        # 4. Overlap Drawing Task (Placeholder for Member 2's drawing code)
        for i, space in enumerate(mock_spaces):
            pts = np.array(space, np.int32).reshape((-1, 1, 2))
            color = (0, 0, 255) if occupancy_states[i] else (0, 255, 0) # Red if full, Green if free
            cv2.polylines(enhanced_frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(enhanced_frame, f"Bay {i+1}", (space[0][0], space[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        available_count = occupancy_states.count(False)
        return enhanced_frame, available_count

    def check_occupancy(self, vehicle_boxes, space_polygons):
        """
        Calculates if a detected vehicle's center point sits inside a parking space polygon.
        """
        states = []
        for polygon in space_polygons:
            space_occupied = False
            poly_array = np.array(polygon, dtype=np.int32)
            
            for box in vehicle_boxes:
                x1, y1, x2, y2 = map(int, box)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Perform an OpenCV polygon point containment test
                is_inside = cv2.pointPolygonTest(poly_array, (center_x, center_y), False)
                if is_inside >= 0:
                    space_occupied = True
                    break
                    
            states.append(space_occupied)
        return states