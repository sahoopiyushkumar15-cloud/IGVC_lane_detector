# IGVC Vision Subsystem Handoff

This directory contains the final, self-contained Vision System for the IGVC autonomous vehicle. 
It encapsulates the Deep Learning UNet, the dynamic color thresholds, the connected components filters, and the Geometric Stable Tracker into a single, easy-to-use Python class.

## Files Included:
- `vision_pipeline.py`: The main API. Contains the `IGVCVisionPipeline` class.
- `model.py`: The PyTorch architecture for the neural network.
- `stable_tracker.py`: The mathematical kalman-style lane boundary tracker.
- `checkpoints/v2_best.pth`: The fully trained weights for the neural network.
- `requirements.txt`: The Python dependencies (PyTorch, OpenCV, NumPy).

## How to use this in the Main Vehicle Loop:

You only need to interact with `vision_pipeline.py`. 

```python
import cv2
from vision_pipeline import IGVCVisionPipeline

# 1. Initialize the system ONCE at startup
vision_system = IGVCVisionPipeline(model_path="checkpoints/v2_best.pth")

# 2. Inside your main ROS or Vehicle loop:
cap = cv2.VideoCapture(0) # Or your ZED/RealSense camera stream

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert BGR camera frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Pass the frame to the vision system
    center_x, mask, left_poly, right_poly = vision_system.process_frame(rgb_frame)
    
    # 4. Use the output for steering!
    # - Pass `center_x` to your Bird's Eye View matrix to calculate real-world steering angles.
    # - Feed the steering angles into your PID controller.
```
