import cv2
import os
import numpy as np
from .court_detector import CourtDetector
from .tracker import Tracker
from .shot_classifier import ShotClassifier
from .tracknet_detector import TrackNetDetector

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
    BALL_SKIP     = 1   # detectar pelota en TODOS los frames (era 2)
    FAR_ZONE_SKIP = 3   # zona lejana amplificada 1 de cada 3 (~3x speedup en far)

    def __init__(self, video_path, output_path, court_config_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.court_detector = CourtDetector(config_path=court_config_path)

        # Obtener posición de la red de la calibración (si existe)
        net_y_fraction = self.court_detector.get_net_y_fraction()

        self.tracker = Tracker(net_y_fraction=net_y_fraction)
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

        # Rastro visual de la pelota (últimas N posiciones) para el vídeo anotado
        self.ball_trail = []
        self.BALL_TRAIL_LEN = 12

        # TrackNetV3 detector (opcional — configurado via env vars)
        self.tracknet = TrackNetDetector()
        self.tracknet_positions = {}  # {frame_idx: (x, y, visible)}

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
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stats["total_video_frames"] = total_frames
        print(f"Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames totales")

        # Adaptar parámetros temporales al FPS real del vídeo.
        # A 60fps se necesita más margen en missed_frames y cooldown escalado.
        fps_ratio = fps / 30.0
        self.shot_classifier.fps = fps
        self.shot_classifier.shot_cooldown = max(20, int(fps * 0.8))  # ~0.8s
        self.tracker.max_lost_frames = int(300 * fps_ratio)           # ~10s
        self.tracker.max_missed_ball_frames = int(self.max_missed_frames * fps_ratio)
        self.max_missed_frames = int(30 * fps_ratio)  # ~0.5s
        
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

        # Pre-procesar con TrackNetV3 si está disponible
        if self.tracknet.available:
            print('[TrackNet] Pre-procesando vídeo para detección de pelota...')
            self.tracknet_positions = self.tracknet.detect_video(
                self.video_path, width, height, on_progress=on_progress
            )
            print(f'[TrackNet] {len(self.tracknet_positions)} frames pre-procesados')

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Líneas de cancha (fijas si hay calibración, sin coste por frame)
            court_lines = self.court_detector.detect(frame)
            if court_lines is not None:
                self.stats["court_detected"] += 1

            # 2. Track personas + detección de pelota
            # Si TrackNetV3 está activo, desactiva YOLO para pelota (run_ball=False)
            using_tracknet = bool(self.tracknet_positions)
            run_ball     = False if using_tracknet else (frame_count % self.BALL_SKIP == 0)
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

            # --- Detección de pelota: TrackNetV3 (preferente) o YOLO (fallback) ---
            trusted_ball_boxes = []
            filtered_ball_boxes = []
            ball_candidates = []

            if using_tracknet:
                tn = self.tracknet_positions.get(frame_count)
                if tn and tn[2]:  # visible=True
                    tx, ty, _ = tn
                    # Filtrar por X del polígono (evita pista adyacente) y por Y con
                    # margen amplio hacia arriba (permite lobs/smashes en el aire pero
                    # descarta detecciones en el techo/fondo de la imagen).
                    poly_xs = court_polygon[:, 0]
                    poly_ys = court_polygon[:, 1]
                    x_min, x_max = float(poly_xs.min()), float(poly_xs.max())
                    # Límite Y superior: parte más alta del polígono menos 60% de margen
                    # (un lob puede subir bastante por encima del fondo de la pista)
                    y_top_court = float(poly_ys.min())
                    y_min_ball = y_top_court * 0.4   # permite zona alta del aire
                    if not (x_min <= tx <= x_max and ty >= y_min_ball):
                        tn = None
                if tn and tn[2]:
                    tx, ty, _ = tn
                    # Filtro de velocidad: rechazar saltos imposibles entre frames.
                    # Pelota a 200km/h en 4K (~10m=3840px) a 30fps → ~700px/frame máx.
                    if self.last_ball_pos is not None:
                        lx, ly, lf = self.last_ball_pos
                        df = max(1, frame_count - lf)
                        jump = ((tx - lx)**2 + (ty - ly)**2) ** 0.5 / df
                        if jump > 750:   # px/frame — imposible físicamente
                            tn = None   # descartar como falso positivo
                if tn and tn[2]:
                    tx, ty, _ = tn
                    r = 8
                    trusted_ball_boxes = [SimpleBox([tx - r, ty - r, tx + r, ty + r])]
                    self.stats["ball_detected"] += 1
                    if self.last_ball_pos is not None:
                        lx, ly, lf = self.last_ball_pos
                        df = frame_count - lf
                        if df > 0:
                            self.ball_velocity = ((tx - lx) / df, (ty - ly) / df)
                    self.last_ball_pos = (tx, ty, frame_count)
                    self.missed_ball_frames = 0
                    self.ball_history.append((tx, ty, frame_count))
                    if len(self.ball_history) > 30:
                        self.ball_history.pop(0)
                else:
                    self.missed_ball_frames += 1

            else:
                # YOLO: construir candidatos y aplicar filtros
                # --- Construir lista unificada de candidatos a pelota ---
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

                # --- Ball candidate extraction + filtrado ---
                for x1, y1, x2, y2, conf in ball_candidates:

                    # Filter 1: ROI - Ignorar el 22% superior (focos a ~15%, techo)
                    if y1 < height * 0.22:
                        continue

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # --- Context-Aware Confidence Logic ---
                    current_threshold = 0.45

                    on_trajectory = False
                    if self.last_ball_pos:
                        lx, ly, lf = self.last_ball_pos
                        dist = ((center_x - lx)**2 + (center_y - ly)**2)**0.5
                        if dist < 120 and (frame_count - lf) < 6:
                            on_trajectory = True

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
                        current_threshold = 0.15
                    elif near_player:
                        current_threshold = 0.22

                    if conf < current_threshold:
                        continue

                    # Filter 2: Size
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_size = max(box_width, box_height)
                    if box_size < 2 or box_size > 55:
                        continue

                    # Filter 3: Aspect ratio
                    aspect_ratio = box_width / box_height if box_height > 0 else 0
                    if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                        continue

                    # Filter: Rechazo de raqueta/mango
                    is_racket = False
                    if near_player_box is not None:
                        px1, py1, px2, py2 = near_player_box
                        inside_player = (px1 < center_x < px2) and (py1 < center_y < py2)
                        if inside_player:
                            ball_speed = np.linalg.norm(self.ball_velocity) if self.ball_velocity else 0
                            if ball_speed < 3.0:
                                is_racket = True
                            aspect_ratio_raw = box_width / box_height if box_height > 0 else 1
                            if aspect_ratio_raw > 3.5 or aspect_ratio_raw < 0.28:
                                is_racket = True
                            if not is_racket:
                                rx1, ry1 = max(0, int(x1)), max(0, int(y1))
                                rx2, ry2 = min(width - 1, int(x2)), min(height - 1, int(y2))
                                if rx2 > rx1 and ry2 > ry1:
                                    patch = frame[ry1:ry2, rx1:rx2]
                                    hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                                    yellow = cv2.inRange(hsv_p, np.array([15, 80, 80]), np.array([40, 255, 255]))
                                    white  = cv2.inRange(hsv_p, np.array([0, 0, 160]),  np.array([180, 50, 255]))
                                    ball_pixels  = cv2.countNonZero(yellow) + cv2.countNonZero(white)
                                    total_pixels = patch.shape[0] * patch.shape[1]
                                    if total_pixels > 0 and ball_pixels / total_pixels < 0.20:
                                        is_racket = True
                    if is_racket:
                        continue

                    # Static check
                    is_static = False
                    for hist_x, hist_y, hist_frame in self.ball_history:
                        distance = ((center_x - hist_x)**2 + (center_y - hist_y)**2)**0.5
                        if distance < self.position_threshold:
                            if frame_count - hist_frame > self.static_threshold:
                                is_static = True
                                break
                    if is_static:
                        continue

                    self.ball_history.append((center_x, center_y, frame_count))
                    if len(self.ball_history) > 30:
                        self.ball_history.pop(0)

                    filtered_ball_boxes.append(
                        (center_x, center_y, SimpleBox([x1, y1, x2, y2]))
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
            # --- fin detección de pelota ---
            
            # El filtro de polígono ya lo aplica tracker.track_frame(),
            # así que todos los boxes en p_boxes son jugadores válidos en pista.
            filtered_p_boxes = list(p_boxes) if p_boxes else []
            
            # --- Shot Classification Logic ---
            ball_pos = None
            if trusted_ball_boxes:
                bx1, by1, bx2, by2 = trusted_ball_boxes[0].xyxy[0].cpu().numpy()
                ball_pos = ((bx1 + bx2)/2, (by1 + by2)/2)

            # Construir player_data antes de detect_impact (nuevo ShotClassifier lo necesita)
            player_data = []
            mapping = person_results.slot_mapping if hasattr(person_results, 'slot_mapping') else {}

            # Debug: ver mapping cada 300 frames
            if frame_count % 300 == 0 and mapping:
                print(f"[DEBUG] Frame {frame_count} - slot_mapping: {mapping}")

            for p_box in filtered_p_boxes:
                px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                yolo_id = int(p_box.id[0]) if p_box.id is not None else -1
                p_id = mapping.get(yolo_id, 0)
                player_data.append((px1, py1, px2, py2, p_id))

            # Solo detectar golpes si los jugadores están inicializados (J1-J4 asignados)
            # y hay al menos un jugador con slot válido (pid > 0) cerca de la pelota
            has_valid_players = any(p[4] > 0 for p in player_data)
            impact = self.shot_classifier.detect_impact(frame_count, ball_pos, player_data) if has_valid_players else None
            if impact:
                event_data = self.shot_classifier.classify_shot(impact, player_data)
                if event_data and event_data.get("player_id", 0) > 0:
                    if event_data["event"] == "Shot":
                        self.detected_shots.append(event_data)
                        self.stats["shots_detected"] += 1
                        if on_event:
                            on_event(event_data)
            # ---------------------------------

            # 3. Draw Annotations
            annotated_frame = frame.copy()
            mapping = person_results.slot_mapping if hasattr(person_results, 'slot_mapping') else {}

            # Slot colors (BGR): J1=azul, J2=cian, J3=amarillo, J4=magenta, 0=gris
            colors = {1: (255, 50, 50), 2: (255, 255, 0), 3: (0, 255, 255), 4: (255, 0, 255), 0: (128, 128, 128)}
            zone_labels = {1: "NEAR", 2: "NEAR", 3: "FAR", 4: "FAR", 0: "?"}

            # --- Línea de red (NET_Y) ---
            net_y_px = int(height * self.tracker.NET_Y)
            cv2.line(annotated_frame, (0, net_y_px), (width, net_y_px), (0, 200, 255), 1)

            # --- Jugadores ---
            # Usar filtered_detections si está disponible (detecciones después del filtro por polígono)
            if hasattr(person_results, 'filtered_detections') and person_results.filtered_detections is not None:
                filtered_detections = person_results.filtered_detections
            else:
                filtered_detections = None

            if frame_count % 300 == 0:
                det_count = len(filtered_detections) if filtered_detections is not None else len(filtered_p_boxes)
                print(f"[DEBUG DRAW] Frame {frame_count}: detections={det_count}, mapping={mapping}")

            drawn_count = 0

            if filtered_detections is not None:
                # Usar detecciones de Supervision (ya filtradas por polígono)
                for i in range(len(filtered_detections)):
                    xyxy = filtered_detections.xyxy[i]
                    px1, py1, px2, py2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    yolo_id = filtered_detections.tracker_id[i] if filtered_detections.tracker_id is not None else -1
                    p_id = mapping.get(int(yolo_id), 0)

                    # Solo dibujar jugadores con IDs válidos (p_id > 0)
                    if p_id == 0:
                        continue

                    drawn_count += 1
                    color = colors.get(p_id, (128, 128, 128))
                    zone  = zone_labels.get(p_id, "?")
                    cv2.rectangle(annotated_frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
                    label = f"J{p_id} [{zone}]"
                    cv2.putText(annotated_frame, label,
                                (int(px1), int(py1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                # Fallback: usar boxes de YOLO (antiguo método)
                for p_box in filtered_p_boxes:
                    px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
                    yolo_id = int(p_box.id[0]) if p_box.id is not None else -1
                    p_id = mapping.get(yolo_id, 0)

                    # Solo dibujar jugadores con IDs válidos (p_id > 0)
                    if p_id == 0:
                        continue

                    drawn_count += 1

                    color = colors.get(p_id, (128, 128, 128))
                    zone  = zone_labels.get(p_id, "?")
                    cv2.rectangle(annotated_frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
                    label = f"J{p_id} [{zone}]"
                    cv2.putText(annotated_frame, label,
                                (int(px1), int(py1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if frame_count % 300 == 0:
                print(f"[DEBUG DRAW] Frame {frame_count}: dibujados={drawn_count} jugadores")

            # --- Rastro de la pelota (trail que se desvanece) ---
            # Si la pelota no se detecta varios frames seguidos, limpiar el trail
            if self.missed_ball_frames > 8:
                self.ball_trail.clear()

            if trusted_ball_boxes:
                bx1, by1, bx2, by2 = trusted_ball_boxes[0].xyxy[0].cpu().numpy()
                ball_cx = int((bx1 + bx2) / 2)
                ball_cy = int((by1 + by2) / 2)
                self.ball_trail.append((ball_cx, ball_cy))
                if len(self.ball_trail) > self.BALL_TRAIL_LEN:
                    self.ball_trail.pop(0)

            for i in range(1, len(self.ball_trail)):
                alpha = i / self.BALL_TRAIL_LEN
                g = int(180 + 75 * alpha)   # verde intensifica al final
                r = int(255 * alpha)        # rojo aparece al final (color cian→blanco)
                thickness = max(1, int(3 * alpha))
                cv2.line(annotated_frame,
                         self.ball_trail[i - 1], self.ball_trail[i],
                         (0, g, r), thickness)

            # Círculo en la posición actual de la pelota
            for box in trusted_ball_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 255), 2)

            # --- Golpes detectados ---
            shot_display_frames = int(fps * 1.0) if 'fps' in dir() else 30
            for shot in self.detected_shots[-5:]:
                if frame_count - shot["frame"] < shot_display_frames:
                    sx, sy = shot["pos"]
                    cv2.circle(annotated_frame, (int(sx), int(sy)), 18, (0, 0, 255), 3)
                    cv2.putText(annotated_frame,
                                f"SHOT J{shot['player_id']}: {shot['shot_type']}",
                                (int(sx) + 22, int(sy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # --- HUD: stats en esquina superior izquierda ---
            hud_lines = [
                f"Frame {frame_count}/{total_frames}",
                f"Shots: {self.stats['shots_detected']}",
                f"Ball det: {self.stats['ball_detected']}",
            ]
            for hi, txt in enumerate(hud_lines):
                cv2.putText(annotated_frame, txt, (12, 28 + hi * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            # --- Draw court lines ---
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
