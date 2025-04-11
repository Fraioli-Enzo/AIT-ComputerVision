# AIT-ComputerVision

## Description  
AIT-ComputerVision is a project designed to detect defects in videos using deep learning models. It uses a pre-trained YOLOv10 model to perform fast and accurate detections.

## Project Structure  
- **models/**: Contains the pre-trained YOLOv10 model (`YOLOv10_smallFDD.torchscript`).  
- **py-src/**: Contains the main script `DefectDetection.py` for running defect detection.  
- **videos/**: Contains sample videos (`video_light.mp4`, `video_nolight.mp4`) to test the program.
- **dxf/**: Contains dxf file with the fabric and the defects (use a dfx viewer to work with them). WORK IN PROGRESS RATIO BETWEEN CONTOUR FABRICS AND DEFECT NOT GOOD

## Requirements  
- Python 3.8 or higher

## Installation  
1. Install the dependencies if needed:  
   ```bash
   pip install -r requirements.txt

## Infos
To see all the research and development of this project, go to the `CameraDetection` repository.
