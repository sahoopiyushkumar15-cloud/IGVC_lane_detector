import cv2
import numpy as np
import sys
from vision_pipeline import IGVCVisionPipeline

def main():
    img_path = r"C:\Users\sahoo\.gemini\antigravity\brain\46542847-0609-471c-a2e8-3953a630557a\.user_uploaded\media_1788102029119.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Loading vision pipeline...")
    pipeline = IGVCVisionPipeline()
    
    print("Processing frame...")
    center_x, mask, l_poly, r_poly = pipeline.process_frame(rgb)
    
    # Create an overlay (Green where the mask is)
    overlay = img.copy()
    overlay[mask == 255] = [0, 255, 0] # Green for the lane boundaries
    
    # Draw polynomials if found
    from stable_tracker import draw_stable_boundary
    draw_stable_boundary(overlay, l_poly, "TRACKED")
    draw_stable_boundary(overlay, r_poly, "TRACKED")
    
    # Draw the target center point
    h, w = img.shape[:2]
    if center_x is not None:
        cv2.circle(overlay, (int(center_x), h - 50), 10, (0, 255, 255), -1) # Yellow dot for steering target
        cv2.putText(overlay, f"Steer Target X: {int(center_x)}", (int(center_x) - 100, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # Save the output
    out_path = "test_image_output.jpg"
    cv2.imwrite(out_path, overlay)
    
    # Also save the raw mask for debugging
    cv2.imwrite("test_image_mask.jpg", mask)
    print("Saved output to test_image_output.jpg and test_image_mask.jpg")

if __name__ == "__main__":
    main()
