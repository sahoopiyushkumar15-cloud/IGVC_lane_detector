import os
import sys
import cv2
import numpy as np
from vision_pipeline import IGVCVisionPipeline

def main():
    # If user provides an image path via CLI: python test_single_image.py my_image.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to the bundled sample image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "sample_track.png")
        
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from '{img_path}'.")
        print("Usage: python test_single_image.py [optional_path_to_image]")
        return
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Initializing vision pipeline and loading pre-trained weights...")
    pipeline = IGVCVisionPipeline()
    
    print("Processing frame...")
    center_x, mask, l_poly, r_poly = pipeline.process_frame(rgb)
    
    print(f"Detected Steering Center X: {center_x}")
    
    # Create an overlay (Green where the mask is)
    overlay = img.copy()
    overlay[mask == 255] = [0, 255, 0] # Green for the lane boundaries
    
    # Draw smoothed boundary lines
    from stable_tracker import draw_stable_boundary
    draw_stable_boundary(overlay, l_poly, "TRACKED")
    draw_stable_boundary(overlay, r_poly, "TRACKED")
    
    # Draw the target center point
    h, w = img.shape[:2]
    if center_x is not None:
        cv2.circle(overlay, (int(center_x), h - 50), 10, (0, 255, 255), -1) # Yellow dot for steering target
        cv2.putText(overlay, f"Steer Target X: {int(center_x)}", (int(center_x) - 100, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # Save the output
    out_overlay = "test_image_output.jpg"
    out_mask = "test_image_mask.jpg"
    cv2.imwrite(out_overlay, overlay)
    cv2.imwrite(out_mask, mask)
    print(f"Done! Saved visualization to '{out_overlay}' and mask to '{out_mask}'.")

if __name__ == "__main__":
    main()
