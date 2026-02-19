import cv2
import os
import numpy as np
from .court_detector import CourtDetector
from .tracker import Tracker
from .shot_classifier import ShotClassifier

class MockTensor:
    def __init__(self, arr): self.arr = arr
    def cpu(self): return self
    def numpy(self): return self.arr

class SimpleBox:
    def __init__(self, xyxy):
        self.xyxy = [MockTensor(np.array(xyxy))]

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.court_detector = CourtDetector()
        self.tracker = Tracker()
        self.shot_classifier = ShotClassifier()
        
        # Metrics
        self.stats = {
            "total_frames": 0,
            "court_detected": 0,
            "ball_detected": 0,
            "players_detected": 0,
            "shots_detected": 0
        }
        self.detected_shots = []  # List of detected shot events
        
        # Temporal filtering for ball detection
        self.ball_history = [] 
        self.static_threshold = 10
        self.position_threshold = 40

        # Ball Interpolation State
        self.last_ball_pos = None # (x, y, frame)
        self.ball_velocity = None # (vx, vy)
        self.missed_ball_frames = 0
        self.max_missed_frames = 15
        self.trajectories = [] # List of lists: [[(x,y,f), ...], ...]

    def process(self, on_event=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file {self.video_path}")
            return

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output writer
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Polígono de cancha fijo para todo el video (filtra jugadores fuera de pista)
        top_y_court = int(height * 0.45)
        court_polygon = np.array([
            [int(width * 0.25), top_y_court],
            [int(width * 0.75), top_y_court],
            [width, height],
            [0, height]
        ], np.int32)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect Court Lines
            court_lines = self.court_detector.detect(frame)
            if court_lines is not None:
                self.stats["court_detected"] += 1

            # 2. Track Objects (solo jugadores dentro de la cancha)
            track_results = self.tracker.track_frame(frame, frame_count, court_polygon=court_polygon)
            person_results = track_results["person_results"]
            ball_results = track_results["ball_results"]
            
            # Update stats
            # Person detections
            p_boxes = person_results.boxes
            if p_boxes and len(p_boxes) > 0:
                self.stats["players_detected"] += 1
            
            # --- Ball candidate extraction ---
            filtered_ball_boxes = []
            trusted_ball_boxes = [] # Initialize here to prevent NameError
            
            if ball_results and ball_results.boxes:
                for box in ball_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter 1: ROI - Ignore top 20% of frame (ceiling area)
                    if y1 < height * 0.2:
                        continue
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # --- Context-Aware Confidence Logic ---
                    current_threshold = 0.45 # Default strict for noise (lights)
                    
                    # 1. Proximity to players
                    near_player = False
                    if p_boxes:
                        for p_box in p_boxes:
                            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                            # Use a generous box around player for sensitivity
                            if (px1 - 80 < center_x < px2 + 80) and (py1 - 80 < center_y < py2 + 80):
                                near_player = True
                                break
                    
                    # 2. Proximity to last known position (temporal continuity)
                    on_trajectory = False
                    if self.last_ball_pos:
                        lx, ly, lf = self.last_ball_pos
                        dist = ((center_x - lx)**2 + (center_y - ly)**2)**0.5
                        # If ball is within 120px of last seen position
                        if dist < 120 and (frame_count - lf) < 6:
                            on_trajectory = True
                    
                    if near_player or on_trajectory:
                        current_threshold = 0.12 # High sensitivity in active zones
                    
                    if conf < current_threshold:
                        continue
                    # ----------------------------------------
                    
                    # Filter 2: Size - Ball should be reasonable (2-55px for blur)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_size = max(box_width, box_height)
                    if box_size < 2 or box_size > 55:
                        continue
                    
                    # Filter 3: Aspect ratio - More tolerant for motion blur
                    aspect_ratio = box_width / box_height if box_height > 0 else 0
                    if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                        continue
                    
                    # Filter 5: Player Adjacency (Racket rejection)
                    is_racket = False
                    if p_boxes:
                        for p_box in p_boxes:
                            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                            p_height = py2 - py1
                            if (px1 - 10 < center_x < px2 + 10) and (py1 - 10 < center_y < py2 - p_height * 0.2):
                                if self.ball_velocity and np.linalg.norm(self.ball_velocity) < 2.5:
                                    is_racket = True
                                    break
                                if center_y < py1 + p_height * 0.6:
                                    is_racket = True
                                    break
                    if is_racket:
                        continue

                    # Static check (temporal noise removal)
                    is_static = False
                    for hist_x, hist_y, hist_frame in self.ball_history:
                        distance = ((center_x - hist_x)**2 + (center_y - hist_y)**2)**0.5
                        if distance < self.position_threshold:
                            if frame_count - hist_frame > self.static_threshold:
                                is_static = True
                                break
                    if is_static: continue
                    
                    self.ball_history.append((center_x, center_y, frame_count))
                    if len(self.ball_history) > 30: self.ball_history.pop(0)
                    
                    filtered_ball_boxes.append((center_x, center_y, box))
                
                # --- Trajectory-Based Validation ---
                validated_ball = self._validate_trajectories([(cx, cy) for cx, cy, b in filtered_ball_boxes], frame_count)
                
                if validated_ball:
                    self.stats["ball_detected"] += 1
                    curr_pos = validated_ball
                    
                    if self.last_ball_pos is not None:
                        lx, ly, lf = self.last_ball_pos
                        df = frame_count - lf
                        if df > 0:
                            self.ball_velocity = ((curr_pos[0] - lx)/df, (curr_pos[1] - ly)/df)
                    
                    self.last_ball_pos = (curr_pos[0], curr_pos[1], frame_count)
                    self.missed_ball_frames = 0
                    
                    # For drawing: use the box associated with this position if available
                    for cx, cy, b in filtered_ball_boxes:
                        if abs(cx - curr_pos[0]) < 1 and abs(cy - curr_pos[1]) < 1:
                            trusted_ball_boxes.append(b)
                            break
                    if not trusted_ball_boxes:
                        # Fallback box if we just have position
                        trusted_ball_boxes.append(SimpleBox([curr_pos[0]-8, curr_pos[1]-8, curr_pos[0]+8, curr_pos[1]+8]))
                else:
                    self.missed_ball_frames += 1
            else:
                self.missed_ball_frames += 1

            # --- Interpolation Logic ---
            if not trusted_ball_boxes and self.last_ball_pos is not None and self.ball_velocity is not None:
                speed = np.linalg.norm(self.ball_velocity)
                if self.missed_ball_frames <= self.max_missed_frames and speed > 3:
                    lx, ly, lf = self.last_ball_pos
                    vx, vy = self.ball_velocity
                    df = frame_count - lf
                    pred_x = lx + vx * df
                    pred_y = ly + vy * df
                    
                    if 0 < pred_x < width and height*0.2 < pred_y < height:
                        filtered_ball_boxes.append(SimpleBox([pred_x-10, pred_y-10, pred_x+10, pred_y+10]))
            # ---------------------------
            
            # El filtro de polígono ya lo aplica tracker.track_frame(),
            # así que todos los boxes en p_boxes son jugadores válidos en pista.
            filtered_p_boxes = list(p_boxes) if p_boxes else []
            
            # --- Shot Classification Logic ---
            ball_pos = None
            if trusted_ball_boxes:
                bx1, by1, bx2, by2 = trusted_ball_boxes[0].xyxy[0].cpu().numpy()
                ball_pos = ((bx1 + bx2)/2, (by1 + by2)/2)
            
            impact = self.shot_classifier.detect_impact(frame_count, ball_pos)
            if impact:
                player_data = []
                mapping = person_results.slot_mapping if hasattr(person_results, 'slot_mapping') else {}
                for p_box in filtered_p_boxes:
                    px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                    yolo_id = int(p_box.id[0]) if p_box.id is not None else -1
                    # Use stable ID from mapping
                    p_id = mapping.get(yolo_id, 0)
                    player_data.append((px1, py1, px2, py2, p_id))
                
                event_data = self.shot_classifier.classify_shot(impact, player_data)
                if event_data:
                    if event_data["event"] == "Shot":
                        self.detected_shots.append(event_data)
                        self.stats["shots_detected"] += 1
                        if on_event:
                            on_event(event_data)
            # ---------------------------------

            # 3. Draw Annotations
            annotated_frame = frame.copy()
            mapping = person_results.slot_mapping if hasattr(person_results, 'slot_mapping') else {}
            
            # Version header
            cv2.putText(annotated_frame, "PADEL STATS PRO - v2.2 (Physics Verified)", (w_frame-550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Slot colors
            colors = {1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 255, 255), 4: (255, 0, 255), 0: (128, 128, 128)}
            
            for p_box in filtered_p_boxes:
                px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                yolo_id = int(p_box.id[0]) if p_box.id is not None else -1
                p_id = mapping.get(yolo_id, 0)
                
                color = colors.get(p_id, (128, 128, 128))
                cv2.rectangle(annotated_frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 3)
                cv2.putText(annotated_frame, f"Jugador {p_id}", (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            for box in trusted_ball_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            for shot in self.detected_shots[-5:]:
                if frame_count - shot["frame"] < 30:
                    sx, sy = shot["pos"]
                    cv2.circle(annotated_frame, (int(sx), int(sy)), 15, (0, 0, 255), 3)
                    cv2.putText(annotated_frame, f"SHOT: {shot['shot_type']} (P{shot['player_id']})", 
                                (int(sx) + 20, int(sy)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw court lines
            annotated_frame = self.court_detector.draw_lines(annotated_frame, court_lines)
            
            # Write frame
            out.write(annotated_frame)
            frame_count += 1
            self.stats["total_frames"] += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            
            if frame_count % 300 == 0:
                cur_total = self.stats["total_frames"] or 1
                print(f"--- Intermediate Stats (Frame {frame_count}) ---")
                print(f"Court: {self.stats['court_detected']/cur_total:.1%}")
                print(f"Ball: {self.stats['ball_detected']/cur_total:.1%}")
                print(f"Shots: {self.stats['shots_detected']}")
                print("---------------------------------------------")

        cap.release()
        out.release()

    def _validate_trajectories(self, candidates, frame_count):
        """
        candidates: List of (x, y) 
        """
        # 1. Update existing trajectories
        matched_traj_indices = set()
        for cand_pos in candidates:
            best_t_idx = -1
            best_dist = 150 # Max pixels per frame
            
            for t_idx, traj in enumerate(self.trajectories):
                if t_idx in matched_traj_indices: continue
                last_pos, last_frame = traj[-1][0], traj[-1][1]
                dt = frame_count - last_frame
                if dt > 5: continue
                
                dist = ((cand_pos[0]-last_pos[0])**2 + (cand_pos[1]-last_pos[1])**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_t_idx = t_idx
            
            if best_t_idx != -1:
                self.trajectories[best_t_idx].append((cand_pos, frame_count))
                matched_traj_indices.add(best_t_idx)
            else:
                # Start new possible trajectory
                self.trajectories.append([(cand_pos, frame_count)])
        
        # 2. Cleanup old trajectories
        self.trajectories = [t for t in self.trajectories if (frame_count - t[-1][1]) < 8]
        
        # 3. Find the best validated trajectory (min length 4)
        best_point = None
        max_len = 0
        for traj in self.trajectories:
            if len(traj) >= 4:
                if len(traj) > max_len:
                    max_len = len(traj)
                    best_point = traj[-1][0]
                    
        return best_point
