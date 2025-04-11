from ultralytics import YOLO
import numpy as np
import ezdxf
import cv2
import os

def initialize_video_capture(video_path_bool):
    if video_path_bool:
        video_path = os.path.join(base_path, "videos\\video_nolight.mp4") 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        return cap
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
        return cap
    
def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return binary_mask, contours

def create_foreground_mask(binary_mask, contours, min_area=150000):
    foreground_mask = np.zeros_like(binary_mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(foreground_mask, [contour], 0, 255, -1)
    return foreground_mask

def check_intersection(contours, middle_line_x, video_height):
    for contour in contours:
        if cv2.pointPolygonTest(contour, (middle_line_x, int(video_height / 2)), False) >= 0:
            return True
    return False

def draw_middle_line(frame, middle_line_x, video_height, intersects_contour, frame_counter, intersection_frame):
    line_color = (0, 255, 0) if intersects_contour else (0, 0, 255)
    cv2.line(frame, (middle_line_x, 0), (middle_line_x, video_height), line_color, 3)
    frame_counter += 1
    if intersects_contour:
        intersection_frame += 1
        print(f"Frame {frame_counter} | Number {intersection_frame}")
    return frame_counter, intersection_frame

def display_foreground(frame, foreground_mask, frame_counter, roi_x1, roi_y1, roi_x2, roi_y2):
    foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    cv2.putText(foreground, f"Frame no{frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(foreground, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(foreground, "YOLO ROI", (roi_x1 + 10, roi_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Foreground", foreground)

def windows_management():
    # Create a window for the control panel
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Control Panel", cv2.WND_PROP_TOPMOST, 1)
    cv2.namedWindow("Foreground", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Foreground", cv2.WND_PROP_TOPMOST, 2)

    # Add a trackbar for the confidence threshold (0 to 100, scaled to 0.0 - 1.0)
    cv2.createTrackbar("Threshold", "Control Panel", 25, 100, lambda x: None)
    cv2.createTrackbar("Saturation", "Control Panel", 100, 100, lambda x: None) # Work in progress for saturation

def draw_bounding_boxes(results, frame, model, frame_counter):
    # Fix for extracting bounding box coordinates
    coordinates = ()  # Initialize coordinates to None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Ensure box.xyxy is flattened into a list of four elements
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Access the first element if it's nested
            confidence = float(box.conf)  # Convert confidence to a float
            class_id = int(box.cls)  # Convert class ID to an integer
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            coordinates = (x1, y1, x2, y2)  # Store the coordinates of the bounding box
    return coordinates
            

def draw_dxf_fabrics(msp, intersection_frame, last_X, coordinates):
    Y = 0  
    X = intersection_frame * 1
    if intersection_frame != 0:
        msp.add_line((last_X, Y), (X, Y))
        msp.add_line((last_X, Y + 100), (X, Y + 100))
    else:
        msp.add_line((X, Y), (X, Y + 100)) 
    if len(coordinates) > 0:
        x1, y1, x2, y2 = coordinates
        msp.add_lwpolyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], close=True)
    last_X = X  
    return last_X, X, Y
 
def detect_fabric_start_end(video_path_bool=False):   
    cap = initialize_video_capture(video_path_bool)
    if cap is None:
        return

    #dxf setup
    doc = ezdxf.new()
    msp = doc.modelspace()
    last_X = 0  # Initialize the last X coordinate for DXF lines

    intersection_frame = 0  # Initialize the test variable
    frame_counter = 0
    frame_delay = 24
    print("Press 'q' to quit")

    model_path = os.path.join(base_path, f"models\\{model_version_epoch}.torchscript")
    model = YOLO(model_path, task="detect")

    windows_management()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Get the current threshold value from the trackbar
        confidence_threshold = cv2.getTrackbarPos("Threshold", "Control Panel") / 100.0
        saturation_value = cv2.getTrackbarPos("Saturation", "Control Panel") / 100.0
    
        # Adjust saturation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
        hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] * saturation_value, 0, 255)  # Adjust saturation
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)  # Convert back to BGR

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        middle_line_x = int(video_width / 2)

        roi_x1, roi_y1, roi_x2, roi_y2 = middle_line_x+3, 0, middle_line_x+403, video_height

        # Crop the frame to the ROI
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        binary_mask, contours = process_frame(frame)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        foreground_mask = create_foreground_mask(binary_mask, contours)
        cv2.line(foreground_mask, (middle_line_x, 0), (middle_line_x, video_height), 255, 3)

        intersects_contour = check_intersection(contours, middle_line_x, video_height)
        frame_counter, intersection_frame = draw_middle_line(frame, middle_line_x, video_height, intersects_contour, frame_counter, intersection_frame)

        results = model(roi_frame)

        # Filter detections based on the confidence threshold
        filtered_results = []
        for result in results:
            filtered_boxes = []
            for box in result.boxes:
                if box.conf >= confidence_threshold:  # Apply the threshold
                    filtered_boxes.append(box)
            result.boxes = filtered_boxes  # Update the result with filtered boxes
            filtered_results.append(result)
        
        coordinates = draw_bounding_boxes(results, roi_frame, model, frame_counter)
        print(len(coordinates))

        last_X, X, Y = draw_dxf_fabrics(msp, intersection_frame, last_X, coordinates)

        display_foreground(frame, foreground_mask, frame_counter, roi_x1, roi_y1, roi_x2, roi_y2)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
    
    # Save the DXF file
    dxf_path = os.path.join(base_path, "dxf\\fabric_detection.dxf")
    msp.add_line((X, Y), (X, Y + 100))
    doc.saveas(dxf_path)
    print(f"DXF file saved at: {dxf_path}")
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    base_path = os.path.dirname("AIT-FabricDetection")

    # Create a dictionary mapping indices to model names
    models = {
        "1": "YOLOv8_smallFDD25",
        "2": "YOLOv10_smallFDD25",
        "3": "YOLOv11_smallFDD25",
        "4": "YOLOv11_smallFDD50",
    } 
    # Set the model_version_epoch based on the selected index
    model_index = "2"
    model_version_epoch = models[model_index]
    print(f"Selected model: {model_version_epoch}")
    detect_fabric_start_end(video_path_bool=True) # Set to True for video file, False for webcam | Default is False
