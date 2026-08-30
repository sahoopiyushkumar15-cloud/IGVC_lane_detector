import numpy as np
import cv2

class StableBoundaryTracker:
    def __init__(self, max_hold=5, band_height=20, max_jump=50):
        self.prev_l = None
        self.prev_r = None
        self.hold_l = 0
        self.hold_r = 0
        self.max_hold = max_hold
        self.band_height = band_height
        self.max_jump = max_jump
        
    def _extract_nodes(self, pts):
        min_y = np.min(pts[:, 0])
        max_y = np.max(pts[:, 0])
        nodes = []
        for y_start in range(min_y, max_y, self.band_height):
            y_end = y_start + self.band_height
            band_pts = pts[(pts[:, 0] >= y_start) & (pts[:, 0] < y_end)]
            if len(band_pts) > 0:
                mean_y = int(np.mean(band_pts[:, 0]))
                mean_x = int(np.mean(band_pts[:, 1]))
                nodes.append((mean_y, mean_x))
        # Ensure we don't return an empty array if band logic misses due to exact bounds
        if len(nodes) == 0:
            nodes.append((int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))))
        return nodes, min_y, max_y
        
    def _compare_to_prev(self, cand_nodes, prev_poly):
        if prev_poly is None or len(prev_poly['nodes']) == 0:
            return 0
        cand_mean_x = np.mean([n[1] for n in cand_nodes])
        prev_mean_x = np.mean([n[1] for n in prev_poly['nodes']])
        return abs(cand_mean_x - prev_mean_x)

    def process_frame(self, or_mask):
        h, w = or_mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(or_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        roi_top, roi_bottom = int(h * 0.4), int(h * 0.95)
        valid_mask = np.zeros_like(cleaned)
        valid_mask[roi_top:roi_bottom, :] = cleaned[roi_top:roi_bottom, :]
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
        
        camera_center = w // 2
        
        left_cands, right_cands = [], []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100: continue
            
            pts = np.column_stack(np.where(labels == i))
            min_y, max_y = np.min(pts[:, 0]), np.max(pts[:, 0])
            if max_y - min_y < 20: continue
            
            mean_x = np.mean(pts[:, 1])
            cand = {'pts': pts, 'min_y': min_y, 'max_y': max_y, 'score': (max_y-min_y)*2 + area*0.5, 'mean_x': mean_x}
            
            dist_l = float('inf')
            dist_r = float('inf')
            
            if self.prev_l is not None:
                prev_l_mean = np.mean([n[1] for n in self.prev_l['nodes']])
                dist_l = abs(mean_x - prev_l_mean)
            if self.prev_r is not None:
                prev_r_mean = np.mean([n[1] for n in self.prev_r['nodes']])
                dist_r = abs(mean_x - prev_r_mean)
                
            if self.prev_l is not None and self.prev_r is not None:
                if dist_l < dist_r:
                    left_cands.append(cand)
                else:
                    right_cands.append(cand)
            else:
                if mean_x < camera_center: left_cands.append(cand)
                else: right_cands.append(cand)
                
        best_l = max(left_cands, key=lambda c: c['score'], default=None)
        best_r = max(right_cands, key=lambda c: c['score'], default=None)
        
        curr_l_poly, curr_r_poly = None, None
        status_l, status_r = "MISSING", "MISSING"
        jump_l, jump_r = 0, 0
        
        if best_l:
            nodes, my, My = self._extract_nodes(best_l['pts'])
            jump_l = self._compare_to_prev(nodes, self.prev_l)
            if jump_l > self.max_jump and self.prev_l is not None:
                if self.hold_l < self.max_hold:
                    curr_l_poly = self.prev_l
                    status_l = "HELD"
                    self.hold_l += 1
            else:
                curr_l_poly = {'nodes': nodes, 'min_y': my, 'max_y': My}
                status_l = "OBSERVED"
                self.hold_l = 0
        else:
            if self.hold_l < self.max_hold and self.prev_l is not None:
                curr_l_poly = self.prev_l
                status_l = "HELD"
                self.hold_l += 1
                
        if best_r:
            nodes, my, My = self._extract_nodes(best_r['pts'])
            jump_r = self._compare_to_prev(nodes, self.prev_r)
            if jump_r > self.max_jump and self.prev_r is not None:
                if self.hold_r < self.max_hold:
                    curr_r_poly = self.prev_r
                    status_r = "HELD"
                    self.hold_r += 1
            else:
                curr_r_poly = {'nodes': nodes, 'min_y': my, 'max_y': My}
                status_r = "OBSERVED"
                self.hold_r = 0
        else:
            if self.hold_r < self.max_hold and self.prev_r is not None:
                curr_r_poly = self.prev_r
                status_r = "HELD"
                self.hold_r += 1

        self.prev_l, self.prev_r = curr_l_poly, curr_r_poly
        return best_l, best_r, curr_l_poly, curr_r_poly, status_l, status_r, jump_l, jump_r

def draw_stable_boundary(overlay, poly, status):
    if poly is None or status == "MISSING" or len(poly['nodes']) < 2: return
    color = (0, 255, 0)
    if status == "HELD": color = (0, 255, 255)
    
    pts = np.array([[n[1], n[0]] for n in poly['nodes']], dtype=np.int32)
    cv2.polylines(overlay, [pts], False, color, 4)
    
def draw_stable_center(overlay, l_poly, r_poly):
    if l_poly is None or r_poly is None or len(l_poly['nodes']) < 2 or len(r_poly['nodes']) < 2: return
    
    l_pts = np.array([[n[0], n[1]] for n in l_poly['nodes']])
    r_pts = np.array([[n[0], n[1]] for n in r_poly['nodes']])
    
    min_y = max(l_poly['min_y'], r_poly['min_y'])
    max_y = min(l_poly['max_y'], r_poly['max_y'])
    
    if max_y <= min_y: return
    
    y_vals = np.linspace(min_y, max_y, max_y - min_y + 1)
    
    # ensure sorted by y for interp
    l_idx = np.argsort(l_pts[:,0])
    r_idx = np.argsort(r_pts[:,0])
    
    l_x = np.interp(y_vals, l_pts[l_idx, 0], l_pts[l_idx, 1])
    r_x = np.interp(y_vals, r_pts[r_idx, 0], r_pts[r_idx, 1])
    
    c_x = (l_x + r_x) / 2
    
    fit_pts = np.column_stack((c_x, y_vals)).astype(np.int32)
    cv2.polylines(overlay, [fit_pts], False, (255, 0, 0), 2)
