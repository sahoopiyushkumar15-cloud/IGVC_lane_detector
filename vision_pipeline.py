import cv2
import numpy as np
import torch
import torchvision.transforms as T
from model import MobileNetV2_UNet
from stable_tracker import StableBoundaryTracker

class IGVCVisionPipeline:
    def __init__(self, model_path="checkpoints/v2_best.pth", device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Load Model
        self.model = MobileNetV2_UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()
        
        # Initialize Tracker
        self.tracker = StableBoundaryTracker(max_jump=50)

    def get_color_mask(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Orange/Red/Yellow Cone and Chalk Mask
        lower_yellow = np.array([0, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White Chalk Mask
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 25, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        return cv2.bitwise_or(yellow_mask, white_mask)

    def process_frame(self, rgb_frame):
        """
        Takes a raw RGB frame from the camera and returns the steering center and mathematical boundaries.
        Returns:
            center_x (float): The target X pixel coordinate for steering (Bird's Eye View needed later).
            clean_mask (numpy array): The final binary segmentation mask.
            l_poly (numpy array): Left lane polynomial coefficients.
            r_poly (numpy array): Right lane polynomial coefficients.
        """
        h, w = rgb_frame.shape[:2]
        
        # 1. Neural Network Inference (Crop to bottom 55% for model)
        crop_h = int(h * 0.45)
        cropped = rgb_frame[crop_h:, :]
        resized = cv2.resize(cropped, (512, 256))
        
        with torch.no_grad():
            tensor_img = self.normalize(self.to_tensor(resized)).unsqueeze(0).to(self.device)
            prob_map = self.model(tensor_img).squeeze().cpu().numpy()
            
        prob_full = np.zeros((h, w), dtype=np.float32)
        prob_full[crop_h:, :] = cv2.resize(prob_map, (w, h - crop_h), interpolation=cv2.INTER_LINEAR)
        unet_mask = (prob_full > 0.10).astype(np.uint8) * 255
        
        # 2. Color Thresholding (Cones + Glare Recovery)
        color_mask = self.get_color_mask(rgb_frame)
        or_mask = cv2.bitwise_or(unet_mask, color_mask)
        
        # 3. Dynamic Hood Masking (Block out vehicle hardware)
        hood_h_start = int(h * 0.7)
        hood_w_start = int(w * 0.25)
        hood_w_end = int(w * 0.75)
        or_mask[max(0, hood_h_start):h, hood_w_start:hood_w_end] = 0
        
        # 4. Connected Components Area Filter (Remove noise < 200px)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(or_mask, connectivity=8)
        clean_mask = np.zeros_like(or_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= 200:
                clean_mask[labels == label] = 255
                
        # 5. Geometric Stable Tracking
        best_l, best_r, l_poly, r_poly, stat_l, stat_r, jl, jr = self.tracker.process_frame(clean_mask)
        
        # Calculate target steering center at the bottom-most visible nodes
        center_x = None
        
        def get_bottom_x(poly):
            if poly is None or len(poly['nodes']) == 0: return None
            # Nodes are (y, x). Find the one with max y.
            bottom_node = max(poly['nodes'], key=lambda n: n[0])
            return bottom_node[1]
            
        left_x = get_bottom_x(l_poly)
        right_x = get_bottom_x(r_poly)
        
        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) / 2
        elif left_x is not None:
            center_x = left_x + 400 
        elif right_x is not None:
            center_x = right_x - 400
            
        return center_x, clean_mask, l_poly, r_poly

if __name__ == "__main__":
    # Example usage for the Robotics Team
    pipeline = IGVCVisionPipeline()
    cap = cv2.VideoCapture("../Video Project 3.mp4")
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_center, final_mask, l_poly, r_poly = pipeline.process_frame(rgb_frame)
        print(f"Vision System Output - Steer towards pixel X: {target_center}")
