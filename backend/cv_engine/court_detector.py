import cv2
import numpy as np

class CourtDetector:
    def __init__(self):
        self.court_lines = None

    def detect(self, frame, debug=False):
        """
        Detects only the 3 main inner court lines (2 horizontal, 1 central vertical).
        """
        h, w = frame.shape[:2]
        
        # 1. Tighter ROI: Only look at the inner court floor area
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # We focus on the bottom 60% of the frame and centered horizontally
        top_y = int(h * 0.55)
        top_left = [int(w * 0.2), top_y]
        top_right = [int(w * 0.8), top_y]
        pts = np.array([top_left, top_right, [w, h], [0, h]], np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)
        
        # 2. HSV Masking for white lines on blue court
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 190]) # High value for white
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        combined_mask = cv2.bitwise_and(white_mask, white_mask, mask=roi_mask)
        
        # 3. Morphology
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        
        # 4. Hough Lines
        lines = cv2.HoughLinesP(
            combined_mask, 
            1, 
            np.pi/180, 
            threshold=80, 
            minLineLength=100, 
            maxLineGap=50
        )
        
        if lines is not None:
            final_lines = []
            
            # 5. Advanced Pruning
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Horizontal Filter (The 2 cross lines)
                if (angle < 10 or angle > 170):
                    # Horizontal lines should be in the middle-ish vertical range
                    if h * 0.6 < (y1+y2)/2 < h * 0.95:
                        final_lines.append(line)
                
                # Vertical Filter (The central service line)
                elif (80 < angle < 100):
                    # Vertical line should be centered horizontally
                    if w * 0.4 < (x1+x2)/2 < w * 0.6:
                        final_lines.append(line)
            
            lines = np.array(final_lines) if final_lines else None
        
        self.court_lines = lines
        return lines

    def draw_lines(self, frame, lines=None):
        if lines is None:
            lines = self.court_lines
            
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green and thicker
        return frame
