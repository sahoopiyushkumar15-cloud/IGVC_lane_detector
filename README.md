# IGVC Vision Subsystem Handoff (Autonomous Lane & Boundary Detection)

This repository contains the complete, self-contained Vision Perception System for the IGVC autonomous vehicle. It combines a PyTorch MobileNetV2 U-Net, dynamic HSV color thresholding (for chalk lines and orange/yellow cones), Connected Components filtering, and a custom Geometric Stable Tracker into a single, high-performance Python class.

---

## ⚡ Zero-Training Deployment (Ready Out-of-the-Box)

> **No additional training is required!**  
> The repository includes fully trained model weights inside `checkpoints/v2_best.pth`.  
> The system runs in **pure inference mode**, meaning the Robotics / Cyber / Controls team can deploy it immediately on the vehicle's onboard computer (NVIDIA Jetson, laptop, or industrial PC) without needing GPUs for training or setting up training datasets.

---

## 📂 Repository File Structure

```text
IGVC_lane_detector/
├── checkpoints/
│   └── v2_best.pth         # Pre-trained deep learning weights (MobileNetV2 U-Net)
├── model.py                # PyTorch architecture definition (U-Net with MobileNetV2 backbone)
├── stable_tracker.py       # Mathematical tracker for smoothing jitter & filtering spurious jumps
├── vision_pipeline.py      # MAIN API: The IGVCVisionPipeline class
├── test_single_image.py    # Test script to verify the pipeline on still images
├── sample_track.png        # Sample test frame from the IGVC course
├── requirements.txt        # Python package dependencies
└── README.md               # Documentation and integration guide
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahoopiyushkumar15-cloud/IGVC_lane_detector.git
   cd IGVC_lane_detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: PyTorch with GPU/CUDA support is recommended for highest FPS, but the pipeline will automatically fall back to CPU if no CUDA GPU is detected.)*

---

## 🧪 Quick Test (Verify in 5 Seconds)

Run the included verification script to test the model immediately:

```bash
python test_single_image.py
```

This will:
1. Load the pre-trained weights from `checkpoints/v2_best.pth`.
2. Process `sample_track.png`.
3. Output the calculated steering target: `Detected Steering Center X: 442.5`.
4. Generate two visual output files in your folder:
   - `test_image_output.jpg`: The camera frame overlaid with detected lane boundaries (green), boundary curves, and a yellow steering target dot.
   - `test_image_mask.jpg`: The raw binary segmentation mask.

To test on your own custom image:
```bash
python test_single_image.py path/to/your_image.jpg
```

---

## 🚗 How to Integrate into the Vehicle Main / ROS Loop

The Robotics / Controls subsystem only needs to interact with `vision_pipeline.py`.

```python
import cv2
from vision_pipeline import IGVCVisionPipeline

# 1. Initialize the pipeline ONCE during system startup
# Automatically selects CUDA GPU if available; otherwise uses CPU
vision = IGVCVisionPipeline(model_path="checkpoints/v2_best.pth")

# 2. Inside your camera capture / ROS subscriber loop:
cap = cv2.VideoCapture(0)  # Or your RealSense / ZED camera feed

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert BGR frame from OpenCV to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run the vision pipeline
    center_x, mask, l_poly, r_poly = vision.process_frame(rgb_frame)
    
    # 3. Use the outputs for vehicle control:
    # - center_x: Pixel X coordinate of the path midpoint at the bottom of the visible track.
    #   Pass center_x through your Bird's Eye View (BEV) Homography matrix to get lateral offset in meters,
    #   then feed into your PID or Pure Pursuit steering controller.
    # - mask: Binary mask for visual debugging or publishing to ROS RViz.
```

---

## 📊 Output Specifications

When calling `vision.process_frame(rgb_frame)`:

| Output | Type | Description | Subsystem Use |
|---|---|---|---|
| `center_x` | `float` or `None` | Midpoint X pixel coordinate between left and right boundaries | Fed into PID / steering control |
| `mask` | `np.ndarray` (uint8) | Cleaned binary mask (255 = lane/cone, 0 = background) | Debugging / ROS RViz feed |
| `l_poly` | `dict` or `None` | Tracked left boundary nodes (`nodes`, `min_y`, `max_y`) | Path planning / boundary checks |
| `r_poly` | `dict` or `None` | Tracked right boundary nodes (`nodes`, `min_y`, `max_y`) | Path planning / boundary checks |

---

## 🛡️ Pipeline Robustness Features Built-in

- **Dynamic Color Recovery:** Handles orange/yellow cones and chalk lines even under direct sun glare.
- **Dynamic Hood Mask:** Automatically eliminates robot chassis/camera hood reflections from confusing the detector.
- **Connected Components Filtering:** Ignores noise clusters smaller than 200 pixels.
- **Geometric Continuity:** Uses memory tracking so that brief camera occlusions or dashed lines do not cause sudden steering jerks.
