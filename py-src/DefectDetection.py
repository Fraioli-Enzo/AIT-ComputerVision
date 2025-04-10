from ultralytics import YOLO
import numpy as np
import cv2
import os

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
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

def draw_middle_line(frame, middle_line_x, video_height, intersects_contour):
    line_color = (0, 255, 0) if intersects_contour else (0, 0, 255)
    cv2.line(frame, (middle_line_x, 0), (middle_line_x, video_height), line_color, 3)

def display_foreground(frame, foreground_mask):
    foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    cv2.imshow("Foreground", foreground)

def detect_fabric_start_end(video_path):
    cap = initialize_video_capture(video_path)
    if cap is None:
        return

    frame_delay = 20
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        middle_line_x = int(video_width / 2)

        binary_mask, contours = process_frame(frame)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        foreground_mask = create_foreground_mask(binary_mask, contours)
        cv2.line(foreground_mask, (middle_line_x, 0), (middle_line_x, video_height), 255, 3)

        intersects_contour = check_intersection(contours, middle_line_x, video_height)
        draw_middle_line(frame, middle_line_x, video_height, intersects_contour)

        display_foreground(frame, foreground_mask)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    base_path = os.path.dirname("AIT-FabricDetection")
    video_path = os.path.join(base_path, "videos\\video_nolight.mp4")

    # Create a dictionary mapping indices to model names
    models = {
        "1": "YOLOv10_smallFDD",
    }
    model_index = "1" 
    
    # Set the model_version_epoch based on the selected index
    model_version_epoch = models[model_index]
    print(f"Selected model: {model_version_epoch}")
    detect_fabric_start_end(video_path)
