import cv2
import numpy as np
from ultralytics import YOLO
# Import Taha's optimized image processing and rendering functions
from utils import enhance_frame, draw_parking_bays, count_available_slots

class ParkingPipeline:
    def __init__(self, custom_model_path="models/best.pt"):
        """
        Initializes the object detection and instance segmentation models.
        Default custom model path looks inside the project's models/ folder.
        """
        # Model 1: Pre-trained out-of-the-box vehicle detector
        self.vehicle_model = YOLO('models/yolov8n.pt') 
        
        # Model 2: Custom fine-tuned instance segmentation model (from Member 3)
        if custom_model_path:
            self.custom_space_model = YOLO(custom_model_path)
        else:
            self.custom_space_model = None

    def process_frame(self, frame):
        """
        Main entry point that the Streamlit UI framework calls inside its processing loop.
        Takes a raw OpenCV BGR frame, processes it, and returns (annotated_frame, available_slots).
        """
        # 1. Image Enhancement (CLAHE Optimization via utils)
        enhanced_frame = enhance_frame(frame) 
        
        # 2. Vehicle Detection (Model 1)
        # Filters for Cars (2), Motorbikes (3), Buses (5), and Trucks (7)
        vehicle_results = self.vehicle_model(enhanced_frame, classes=[2, 3, 5, 7], verbose=False)
        
        # Extract vehicle bounding boxes [[x1, y1, x2, y2], ...]
        vehicle_boxes = []
        if len(vehicle_results) > 0 and len(vehicle_results[0].boxes) > 0:
            vehicle_boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
        
        # 3. Dynamic Space Tracking Logic via Instance Segmentation (Model 2)
        detected_spaces = []
        if self.custom_space_model is not None:
            # Run inference using the custom model to extract parking slot masks
            space_results = self.custom_space_model(enhanced_frame, verbose=False)
            
            # Check if any segmentation masks were successfully parsed
            if len(space_results) > 0 and space_results[0].masks is not None:
                # .xy extracts the raw multi-point polygon coordinates
                detected_spaces = [poly.astype(np.int32) for poly in space_results[0].masks.xy]
        
        # Fallback to standard mock spaces if the model does not return layout shapes
        if len(detected_spaces) == 0:
            detected_spaces = [np.array(poly, np.int32) for poly in [
                [[50, 300], [200, 300], [200, 450], [50, 450]],
                [[250, 300], [400, 300], [400, 450], [250, 450]],
                [[450, 300], [600, 300], [600, 450], [450, 450]]
            ]]
        
        # 4. Calculate Spatial Occupancy Matrix Status using vector geometry math
        occupancy_states = self.check_occupancy(vehicle_boxes, detected_spaces)
        
        # 5. Render Translucent Visual Overlays (Alpha-Blending via utils)
        annotated_frame = draw_parking_bays(enhanced_frame, detected_spaces, occupancy_states)
        
        # 6. Sum up final availability metrics
        available_count = count_available_slots(occupancy_states)
        
        return annotated_frame, available_count

    def check_occupancy(self, vehicle_boxes, space_polygons):
        """
        Evaluates whether a detected vehicle's center pixel point drops inside a space polygon.
        """
        states = []
        for polygon in space_polygons:
            space_occupied = False
            poly_array = np.array(polygon, dtype=np.int32)
            
            for box in vehicle_boxes:
                x1, y1, x2, y2 = map(int, box)
                # Compute the midpoints of the vehicle's bounding envelope
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Run the compiled OpenCV contour point-containment test
                is_inside = cv2.pointPolygonTest(poly_array, (center_x, center_y), False)
                
                # A score >= 0 indicates the point is safely inside or on the perimeter
                if is_inside >= 0:
                    space_occupied = True
                    break
                    
            states.append(space_occupied)
        return states