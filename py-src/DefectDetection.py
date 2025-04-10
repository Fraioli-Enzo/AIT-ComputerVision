from ultralytics import YOLO
import numpy as np
import cv2
import os

def detect_fabric_start_end(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Initialize delay between frames (in milliseconds)
    frame_delay = 20 
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Put a line in the middle
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.line(frame, (int(video_width / 2), 0), (int(video_width / 2), video_height), (0, 255, 0), 3)

        # Apply simple background removal using grayscale thresholding
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        # Create a mask for the foreground
        foreground_mask = np.zeros_like(binary_mask)
        for contour in contours:
            # Only keep larger contours (adjust area threshold as needed)
            if cv2.contourArea(contour) > 150000:
                cv2.drawContours(foreground_mask, [contour], 0, 255, -1)
        
        # Always ensure the middle vertical line is visible in the mask
        cv2.line(foreground_mask, (int(video_width / 2), 0), (int(video_width / 2), video_height), 255, 3)

        # Apply the mask to the original frame to extract foreground
        foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)

        # Display the masked result in a separate window
        cv2.imshow("Foreground", foreground)

        # Wait for the specified delay and check for key presses
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):  # Quit
            break
        
    cap.release()
    cv2.destroyAllWindows()

    return None  



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
