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
    # Frecuencia de muestreo para detección de pelota.
    # La pelota se detecta cada BALL_SKIP frames; entre medias se usa interpolación.
    # Person tracking corre cada frame para mantener continuidad de IDs.
    BALL_SKIP     = 2   # detectar pelota 1 de cada 2 frames  (~2x speedup en ball)
    FAR_ZONE_SKIP = 3   # zona lejana amplificada 1 de cada 3 (~3x speedup en far)

    def __init__(self, video_path, output_path, court_config_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.court_detector = CourtDetector(config_path=court_config_path)
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
        self.max_missed_frames = 30  # 15→30: cubre 0.5s a 60fps (antes 0.25s)
        self.trajectories = [] # List of lists: [[(x,y,f), ...], ...]

    def process(self, on_event=None, on_progress=None):
        """
        on_event(event_data)        — callback por cada golpe detectado
        on_progress(current, total) — callback cada 60 frames con el progreso
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file {self.video_path}")
            return

        # Video properties
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stats["total_video_frames"] = total_frames
        print(f"Video: {width}x{height} @ {fps}fps — {total_frames} frames totales")
        
        # Output writer
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Polígono de cancha: usa calibración del CourtDetector si está disponible,
        # si no, calcula un trapecio de respaldo con el primer frame.
        ret0, first_frame = cap.read()
        if not ret0:
            cap.release()
            out.release()
            return
        court_polygon = self.court_detector.detect_court_polygon(first_frame)
        if court_polygon is None:
            top_y_court = int(height * 0.45)
            court_polygon = np.array([
                [int(width * 0.25), top_y_court],
                [int(width * 0.75), top_y_court],
                [width, height],
                [0, height]
            ], np.int32)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rebobinar al inicio

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Líneas de cancha (fijas si hay calibración, sin coste por frame)
            court_lines = self.court_detector.detect(frame)
            if court_lines is not None:
                self.stats["court_detected"] += 1

            # 2. Track + detección de pelota con frame skipping
            run_ball     = (frame_count % self.BALL_SKIP     == 0)
            run_far_zone = (frame_count % self.FAR_ZONE_SKIP == 0)
            track_results  = self.tracker.track_frame(
                frame, frame_count,
                court_polygon=court_polygon,
                run_ball=run_ball,
                run_far_zone=run_far_zone,
            )
            person_results = track_results["person_results"]
            ball_results   = track_results["ball_results"]       # None si skipped
            ball_far       = track_results.get("ball_far_detections", [])

            # Update stats
            # Person detections
            p_boxes = person_results.boxes
            if p_boxes and len(p_boxes) > 0:
                self.stats["players_detected"] += 1

            # --- Construir lista unificada de candidatos a pelota ---
            # ball_results puede ser None si el frame fue skipped
            ball_candidates = []  # [(x1, y1, x2, y2, conf)]
            if ball_results is not None and ball_results.boxes:
                for box in ball_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    ball_candidates.append((x1, y1, x2, y2, conf))
            # Añadir detecciones de la zona lejana (sin duplicar por IoU > 0.3)
            for fx1, fy1, fx2, fy2, fconf in ball_far:
                duplicate = False
                for cx1, cy1, cx2, cy2, _ in ball_candidates:
                    ix = max(0, min(fx2, cx2) - max(fx1, cx1))
                    iy = max(0, min(fy2, cy2) - max(fy1, cy1))
                    if ix * iy > 0:
                        area_f = (fx2 - fx1) * (fy2 - fy1)
                        area_c = (cx2 - cx1) * (cy2 - cy1)
                        iou = (ix * iy) / (area_f + area_c - ix * iy + 1e-6)
                        if iou > 0.3:
                            duplicate = True
                            break
                if not duplicate:
                    ball_candidates.append((fx1, fy1, fx2, fy2, fconf))

            # --- Ball candidate extraction ---
            filtered_ball_boxes = []
            trusted_ball_boxes = []

            for x1, y1, x2, y2, conf in ball_candidates:
                    
                    # Filter 1: ROI - Ignorar el 22% superior (focos a ~15%, techo)
                    # No subir más: la pelota en el fondo aparece desde ~21% del frame
                    if y1 < height * 0.22:
                        continue
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # --- Context-Aware Confidence Logic ---
                    current_threshold = 0.45  # Estricto por defecto (focos, reflejos)

                    # 1. Continuidad temporal: cerca de última posición conocida
                    on_trajectory = False
                    if self.last_ball_pos:
                        lx, ly, lf = self.last_ball_pos
                        dist = ((center_x - lx)**2 + (center_y - ly)**2)**0.5
                        if dist < 120 and (frame_count - lf) < 6:
                            on_trajectory = True

                    # 2. Proximidad a jugadores — con margen reducido (antes 80px)
                    near_player = False
                    near_player_box = None
                    if p_boxes:
                        for p_box in p_boxes:
                            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                            if (px1 - 50 < center_x < px2 + 50) and (py1 - 50 < center_y < py2 + 50):
                                near_player = True
                                near_player_box = (px1, py1, px2, py2)
                                break

                    if on_trajectory:
                        current_threshold = 0.15   # Seguimiento de trayectoria
                    elif near_player:
                        current_threshold = 0.22   # Cerca de jugador (antes 0.12)

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
                    
                    # Filter: Rechazo de raqueta/mango
                    # Si la detección está DENTRO del bbox del jugador (sin margen),
                    # comprobamos color HSV: una pelota de pádel es blanca o amarilla
                    # y redondeada. Un mango amarillo tiene aspecto ratio elongado.
                    is_racket = False
                    if near_player_box is not None:
                        px1, py1, px2, py2 = near_player_box
                        p_height = py2 - py1
                        inside_player = (px1 < center_x < px2) and (py1 < center_y < py2)
                        if inside_player:
                            # Raqueta/mango: detectada dentro del jugador y pelota lenta
                            ball_speed = np.linalg.norm(self.ball_velocity) if self.ball_velocity else 0
                            if ball_speed < 3.0:
                                is_racket = True
                            # Aspecto ratio muy elongado → mango, no pelota
                            aspect_ratio_raw = box_width / box_height if box_height > 0 else 1
                            if aspect_ratio_raw > 3.5 or aspect_ratio_raw < 0.28:
                                is_racket = True
                            # Confirmación de color: verificar que hay píxeles
                            # amarillos o blancos en la zona detectada
                            if not is_racket:
                                rx1, ry1 = max(0, int(x1)), max(0, int(y1))
                                rx2, ry2 = min(width - 1, int(x2)), min(height - 1, int(y2))
                                if rx2 > rx1 and ry2 > ry1:
                                    patch = frame[ry1:ry2, rx1:rx2]
                                    hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                                    # Amarillo: H 15-40, S>80, V>80
                                    yellow = cv2.inRange(hsv_p,
                                                         np.array([15, 80, 80]),
                                                         np.array([40, 255, 255]))
                                    # Blanco: S<50, V>160
                                    white = cv2.inRange(hsv_p,
                                                        np.array([0, 0, 160]),
                                                        np.array([180, 50, 255]))
                                    ball_pixels = cv2.countNonZero(yellow) + cv2.countNonZero(white)
                                    total_pixels = patch.shape[0] * patch.shape[1]
                                    # Menos del 20% de píxeles de color pelota → no es pelota
                                    if total_pixels > 0 and ball_pixels / total_pixels < 0.20:
                                        is_racket = True
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
                    if len(self.ball_history) > 30:
                        self.ball_history.pop(0)

                    # Guardar como SimpleBox usando las coords ya disponibles
                    filtered_ball_boxes.append(
                        (center_x, center_y,
                         SimpleBox([x1, y1, x2, y2]))
                    )

            # --- Trajectory-Based Validation ---
            if ball_candidates:
                validated_ball = self._validate_trajectories(
                    [(cx, cy) for cx, cy, b in filtered_ball_boxes], frame_count
                )

                if validated_ball:
                    self.stats["ball_detected"] += 1
                    curr_pos = validated_ball

                    if self.last_ball_pos is not None:
                        lx, ly, lf = self.last_ball_pos
                        df = frame_count - lf
                        if df > 0:
                            self.ball_velocity = ((curr_pos[0] - lx) / df,
                                                  (curr_pos[1] - ly) / df)

                    self.last_ball_pos = (curr_pos[0], curr_pos[1], frame_count)
                    self.missed_ball_frames = 0

                    for cx, cy, b in filtered_ball_boxes:
                        if abs(cx - curr_pos[0]) < 1 and abs(cy - curr_pos[1]) < 1:
                            trusted_ball_boxes.append(b)
                            break
                    if not trusted_ball_boxes:
                        trusted_ball_boxes.append(
                            SimpleBox([curr_pos[0]-8, curr_pos[1]-8,
                                       curr_pos[0]+8, curr_pos[1]+8])
                        )
                else:
                    self.missed_ball_frames += 1
            else:
                self.missed_ball_frames += 1

            # --- Interpolation Logic ---
            # Bug fix: la posición interpolada debe ir a trusted_ball_boxes
            # (antes iba a filtered_ball_boxes y nunca llegaba a shot detection)
            if not trusted_ball_boxes and self.last_ball_pos is not None and self.ball_velocity is not None:
                speed = np.linalg.norm(self.ball_velocity)
                if self.missed_ball_frames <= self.max_missed_frames and speed > 1.0:
                    lx, ly, lf = self.last_ball_pos
                    vx, vy = self.ball_velocity
                    df = frame_count - lf
                    pred_x = lx + vx * df
                    pred_y = ly + vy * df

                    if 0 < pred_x < width and height * 0.2 < pred_y < height:
                        trusted_ball_boxes.append(
                            SimpleBox([pred_x - 10, pred_y - 10, pred_x + 10, pred_y + 10])
                        )
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
                    if event_data["event"] == "Shot" and event_data.get("player_id", 0) > 0:
                        self.detected_shots.append(event_data)
                        self.stats["shots_detected"] += 1
                        if on_event:
                            on_event(event_data)
            # ---------------------------------

            # 3. Draw Annotations
            annotated_frame = frame.copy()
            mapping = person_results.slot_mapping if hasattr(person_results, 'slot_mapping') else {}
            
            # Version header
            cv2.putText(annotated_frame, "PADEL STATS PRO - v2.2 (Physics Verified)", (width - 550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
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
            if frame_count % 60 == 0:
                print(f"  {frame_count}/{total_frames} frames procesados…")
                if on_progress:
                    on_progress(frame_count, total_frames)
            
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
                if dt > 10: continue  # 5→10: tolera gaps de 0.17s a 60fps
                
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
        
        # 2. Cleanup old trajectories (alineado con el dt máximo de 10)
        self.trajectories = [t for t in self.trajectories if (frame_count - t[-1][1]) < 12]
        
        # 3. Find the best validated trajectory (min length 3)
        # A 60fps, 4 frames = 67ms. Bajamos a 3 para detectar pelotas rápidas.
        best_point = None
        max_len = 0
        for traj in self.trajectories:
            if len(traj) >= 3:
                if len(traj) > max_len:
                    max_len = len(traj)
                    best_point = traj[-1][0]

        return best_point
