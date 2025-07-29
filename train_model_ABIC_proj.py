#YoloModel Approach - With Deep Learning
from ultralytics import YOLO
import os
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    print("Loaded config:", config)
    
# Load a pretrained YOLOv8 model
print("Starting training...")
if not os.path.exists("yolov8n.pt"):
    print("Error: yolov8n.pt not found")
    exit(1)
if not os.path.exists("config.yaml"):
    print("Error: config.yaml not found")
    exit(1)

model = YOLO("yolov8n.pt")
print("Model loaded, starting training...")

# Train the model on a custom dataset
model.train(
    data="/Users/karthik/Desktop/ABIC - Sorter Project/config.yaml",
    epochs=100,
    imgsz=640,
    batch=1,
    name="organoid_detector",
    verbose=True
)