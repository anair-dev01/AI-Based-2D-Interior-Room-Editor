# AI-Based-2D-Interior-Room-Editor

Overview
AI Interior Editor is a Streamlit-based application that enables object-level editing of interior images using YOLOv8 segmentation and OpenCV.

Requirements
Python 3.9 or higher
Required packages listed in requirements.txt
YOLOv8 segmentation model file yolov8n-seg.pt
Furniture PNG assets organized by category inside the assets folder

Setup
Clone the repository
Install dependencies using pip install -r requirements.txt
Place the model file in the project root directory
Ensure asset folders contain transparent PNG images

Run
Start the application using streamlit run app.py
Upload a room image
Click Detect Furniture to generate segmentation results
Select an object to remove or replace

Notes
The original image remains unchanged during editing
All processing runs locally without cloud services
