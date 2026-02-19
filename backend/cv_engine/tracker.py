from __future__ import annotations

from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    """
    4-slot player tracker con zonas de cancha y asignación óptima.

    Asignación de slots:
      - Slots 1 & 2: zona CERCANA (Y alto en imagen = cerca de la cámara)
        - Slot 1: izquierda, Slot 2: derecha
      - Slots 3 & 4: zona LEJANA (Y bajo en imagen = lejos de la cámara)
        - Slot 3: izquierda, Slot 4: derecha

    Esto garantiza que los equipos nunca intercambien IDs entre sí.

    La separación usa los PIES del jugador (xyxy[3]) comparados con NET_Y,
    que representa la posición de la red en el frame. Los pies de los
    jugadores del fondo nunca cruzan la red, por lo que este límite es fiable.
    """

    # Posición de la red como fracción de la altura del frame.
    # Los pies de los jugadores lejanos siempre quedan por encima de este valor.
    # Ajustar según el ángulo de cámara de tu cancha (típico cámara fondo: 0.60-0.68).
    NET_Y = 0.63

    # Buffer alrededor de la red (fracción de altura).
    # Un jugador dentro del buffer se considera "zona ambigua": no se penaliza
    # por cruzar zona, evitando swaps cuando se acerca a la red.
    NET_Y_BUFFER = 0.05  # ±54px en 1080p

    def __init__(self, model_path='yolov8n.pt', ball_model_path='backend/tennis_ball_best.pt'):
        self.model_persons = YOLO(model_path)
        self.model_persons.to('mps')

        self.model_ball = YOLO(ball_model_path)
        self.model_ball.to('mps')

        # Estado de los 4 slots
        # Cada slot: {"last_pos": (x,y), "last_frame": int, "yolo_id": int, "hist": ndarray, "zone": str}
        self.slots = {1: None, 2: None, 3: None, 4: None}

        # Zona actual de cada slot: near (cámara) o far (fondo).
        # Se inicializa con la distribución por defecto pero se adapta automáticamente
        # si un jugador lleva suficientes frames consecutivos en la zona contraria
        # (p.ej. cambio de lado entre sets).
        self.slot_zones = {1: "near", 2: "near", 3: "far", 4: "far"}

        # Contador de frames consecutivos en zona "equivocada" por slot.
        # Cuando supera ZONE_FLIP_FRAMES, se confirma el cambio de zona.
        self._zone_wrong_frames = {1: 0, 2: 0, 3: 0, 4: 0}
        # Frames consecutivos necesarios para confirmar un cambio de zona.
        # ~2s a 60fps: cubre cambio de lado pero no aproximaciones a la red.
        self.ZONE_FLIP_FRAMES = 120

        self.max_lost_frames = 300   # ~10 segundos a 30fps
        self.max_distance = 250      # píxeles máximos entre frames consecutivos

        # Peso del histograma nuevo en la media exponencial (EWA)
        # Bajo = memoria larga, conservador. Alto = adaptación rápida.
        self.hist_alpha = 0.25

        self.frame_height = None  # Se asigna al primer frame

    # ------------------------------------------------------------------
    # Utilidades privadas
    # ------------------------------------------------------------------

    def _get_zone(self, feet_y: float) -> str:
        """
        Clasifica un jugador como 'near', 'far' o 'ambiguous'.
        'near':      pies claramente por debajo de la red
        'far':       pies claramente por encima de la red
        'ambiguous': pies dentro del buffer ±NET_Y_BUFFER alrededor de la red
                     → no se aplica penalización de zona en el matching
        """
        if self.frame_height is None:
            return "near"
        net_px = self.frame_height * self.NET_Y
        buf_px = self.frame_height * self.NET_Y_BUFFER
        if feet_y > net_px + buf_px:
            return "near"
        if feet_y < net_px - buf_px:
            return "far"
        return "ambiguous"

    def _get_color_histogram(self, frame, xyxy) -> np.ndarray | None:
        """Histograma HSV normalizado de la región del torso del jugador."""
        x1, y1, x2, y2 = map(int, xyxy)
        h_box, w_box = y2 - y1, x2 - x1
        if h_box <= 0 or w_box <= 0:
            return None

        # Torso: 20-70% de la altura, 20-80% del ancho (evita cabeza y piernas)
        crop = frame[
            y1 + int(h_box * 0.20): y1 + int(h_box * 0.70),
            x1 + int(w_box * 0.20): x1 + int(w_box * 0.80)
        ]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Más bins en Hue para mejor discriminación de colores
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _blend_hist(self, old_hist, new_hist) -> np.ndarray | None:
        """Promedia el histograma nuevo con el histórico (EWA)."""
        if old_hist is None:
            return new_hist
        if new_hist is None:
            return old_hist
        blended = (1.0 - self.hist_alpha) * old_hist + self.hist_alpha * new_hist
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended

    def _maybe_flip_zone(self, sid: int, feet_y: float):
        """
        Actualiza el contador de frames en zona incorrecta para el slot sid.
        Si el jugador lleva ZONE_FLIP_FRAMES consecutivos claramente en la zona
        contraria (fuera del buffer), confirma el cambio y actualiza slot_zones.
        Esto permite detectar cambios de lado entre sets sin configuración manual.
        """
        if self.frame_height is None:
            return
        net_px = self.frame_height * self.NET_Y
        buf_px = self.frame_height * self.NET_Y_BUFFER

        if feet_y > net_px + buf_px:
            raw_zone = "near"
        elif feet_y < net_px - buf_px:
            raw_zone = "far"
        else:
            # En el buffer: zona ambigua, no contabilizar en ninguna dirección
            self._zone_wrong_frames[sid] = 0
            return

        if raw_zone != self.slot_zones[sid]:
            self._zone_wrong_frames[sid] += 1
            if self._zone_wrong_frames[sid] >= self.ZONE_FLIP_FRAMES:
                old = self.slot_zones[sid]
                self.slot_zones[sid] = raw_zone
                self._zone_wrong_frames[sid] = 0
                print(f"[Tracker] Slot {sid} cambia zona {old}→{raw_zone} "
                      f"(cambio de lado detectado)")
        else:
            self._zone_wrong_frames[sid] = 0

    def _hist_distance(self, h1, h2) -> float:
        """Chi-cuadrado normalizado entre dos histogramas. 0 = idénticos, 1 = máximos."""
        if h1 is None or h2 is None:
            return 0.5  # Neutro si no hay datos
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
        return min(dist, 2.0) / 2.0

    # ------------------------------------------------------------------
    # Lógica de asignación
    # ------------------------------------------------------------------

    def _update_slots(self, detections: list, frame_count: int, frame=None) -> dict:
        """
        Actualiza los slots con las detecciones del frame actual.
        Retorna {yolo_id: slot_id}.

        Estrategia de dos fases:
        1. Matching global (sin restricción de zona) contra slots ya activos.
           Esto evita que los jugadores pierdan su slot al cruzar NET_Y.
        2. Detecciones sin asignar llenan slots vacíos, usando zona para
           decidir cuál slot vacío corresponde (near vs far).
        """
        if self.frame_height is None and frame is not None:
            self.frame_height = frame.shape[0]

        # Calcular histogramas y zona inicial para cada detección
        for d in detections:
            d["hist"] = self._get_color_histogram(frame, d["xyxy"]) if frame is not None else None
            d["zone"] = self._get_zone(d["xyxy"][3])  # pies = y2

        mapping = {}
        assigned_det_indices = set()

        # --- Fase 1: Matching global contra slots activos ---
        # Se permite cruce de zona pero se penaliza para evitar swaps accidentales.
        # Un jugador near nunca debería costar menos en un slot far que en el suyo.
        CROSS_ZONE_PENALTY = 0.50

        active_slots = [sid for sid, data in self.slots.items() if data is not None]

        if active_slots and detections:
            n_dets  = len(detections)
            n_slots = len(active_slots)
            cost = np.full((n_dets, n_slots), fill_value=2.0)

            for r, d in enumerate(detections):
                for c, sid in enumerate(active_slots):
                    data = self.slots[sid]
                    dist = np.hypot(
                        d["pos"][0] - data["last_pos"][0],
                        d["pos"][1] - data["last_pos"][1]
                    )
                    if dist > self.max_distance * 2.0:
                        continue  # deja cost en 2.0 (rechazado)

                    spatial = min(dist / self.max_distance, 1.0)
                    color   = self._hist_distance(d["hist"], data.get("hist"))
                    base    = 0.55 * color + 0.45 * spatial

                    # Penalizar si la zona del jugador no coincide con la del slot.
                    # Si el jugador está en la zona buffer ("ambiguous"), no se penaliza:
                    # que color+posición decidan sin sesgo de zona.
                    det_zone = d["zone"]
                    slot_zone = self.slot_zones[sid]
                    if det_zone != "ambiguous" and det_zone != slot_zone:
                        penalty = CROSS_ZONE_PENALTY
                    else:
                        penalty = 0.0
                    cost[r, c] = min(base + penalty, 2.0)

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 0.75:
                    continue
                sid = active_slots[c]
                d   = detections[r]
                # Actualizar zona adaptativa antes de guardar en el slot
                self._maybe_flip_zone(sid, d["xyxy"][3])
                self.slots[sid] = {
                    "last_pos":   d["pos"],
                    "last_frame": frame_count,
                    "yolo_id":    d["id"],
                    "hist":       self._blend_hist(self.slots[sid].get("hist"), d["hist"]),
                    "zone":       self.slot_zones[sid],
                }
                mapping[d["id"]] = sid
                assigned_det_indices.add(r)

        # --- Fase 2: Detecciones sin asignar → llenan slots vacíos por zona ---
        unassigned = [(r, d) for r, d in enumerate(detections) if r not in assigned_det_indices]
        unassigned.sort(key=lambda rd: rd[1]["pos"][0])  # izquierda primero

        for r, d in unassigned:
            det_zone = d["zone"]
            if det_zone == "ambiguous":
                # En la zona buffer: elegir el slot vacío más cercano espacialmente
                empty_slots = [(sid, data) for sid, data in self.slots.items() if data is None]
                # Preferir el slot cuya zona coincide con la última posición del jugador
                # (near vs far) usando NET_Y como referencia dura como desempate
                raw_zone = "near" if d["xyxy"][3] > (self.frame_height or 0) * self.NET_Y else "far"
                candidates = [sid for sid, _ in empty_slots if self.slot_zones[sid] == raw_zone]
                if not candidates:
                    candidates = [sid for sid, _ in empty_slots]
            else:
                # Zona clara: buscar slot vacío de la zona correcta
                candidates = [
                    sid for sid, data in self.slots.items()
                    if data is None and self.slot_zones[sid] == det_zone
                ]
                if not candidates:
                    # Fallback: cualquier slot vacío
                    candidates = [sid for sid, data in self.slots.items() if data is None]
            if not candidates:
                continue

            sid = candidates[0]
            self.slots[sid] = {
                "last_pos":   d["pos"],
                "last_frame": frame_count,
                "yolo_id":    d["id"],
                "hist":       d["hist"],
                "zone":       self.slot_zones[sid],
            }
            mapping[d["id"]] = sid

        # Expirar slots sin actualización reciente
        for sid, data in self.slots.items():
            if data and (frame_count - data["last_frame"]) > self.max_lost_frames:
                self.slots[sid] = None

        return mapping

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def track_frame(self, frame, frame_count: int = 0,
                    court_polygon: np.ndarray | None = None,
                    run_ball: bool = True,
                    run_far_zone: bool = True) -> dict:
        """
        court_polygon : polígono np.int32 (N,2) que define la zona de juego.
        run_ball      : ejecutar detección de pelota en este frame (frame skipping).
        run_far_zone  : ejecutar el segundo pase de zona lejana (más costoso).
        """
        # Detección de personas — siempre, necesario para continuidad de IDs
        person_results = self.model_persons.track(
            frame, persist=True, classes=[0], conf=0.25, verbose=False
        )[0]

        mapping = {}
        if person_results.boxes and person_results.boxes.id is not None:
            detections = []
            for i, yolo_id in enumerate(person_results.boxes.id.cpu().numpy()):
                xyxy = person_results.boxes.xyxy[i].cpu().numpy()
                if court_polygon is not None:
                    feet = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
                    if cv2.pointPolygonTest(court_polygon, feet, False) < 0:
                        continue
                pos = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                detections.append({"id": int(yolo_id), "pos": pos, "xyxy": xyxy})

            mapping = self._update_slots(detections, frame_count, frame)

        person_results.slot_mapping = mapping

        # --- Detección de pelota con frame skipping ---
        if run_ball:
            ball_results = self.model_ball.track(
                frame, persist=True, conf=0.10, verbose=False
            )[0]
        else:
            ball_results = None

        if run_far_zone and run_ball:
            ball_far_detections = self._detect_ball_far_zone(frame)
        else:
            ball_far_detections = []

        return {
            "person_results":     person_results,
            "ball_results":       ball_results,
            "ball_far_detections": ball_far_detections,
        }

    # Fracción del frame que se ignora por arriba (focos, techo).
    # Debe coincidir con el filtro ROI usado en processor.py / test_tracker.py.
    ROI_TOP = 0.22

    def _detect_ball_far_zone(self, frame) -> list:
        """
        Segundo pase de detección de pelota sobre la zona lejana de la cancha
        (por encima de la red) ampliada 2× para mejorar la detección de pelotas
        pequeñas. Devuelve lista de (x1, y1, x2, y2, conf) en coordenadas
        originales del frame.

        El crop empieza en ROI_TOP (22 %) para excluir los focos del techo
        antes de ampliar la imagen, evitando falsos positivos.
        """
        if self.frame_height is None:
            return []

        roi_top = int(self.frame_height * self.ROI_TOP)
        net_y = int(self.frame_height * self.NET_Y)
        # Añadir 15% extra por debajo de la red para no perder pelotas en vuelo
        crop_bottom = min(int(net_y * 1.15), self.frame_height)

        crop = frame[roi_top:crop_bottom, :]
        if crop.shape[0] < 40:
            return []

        # Ampliar 2× → la pelota de 3-5px pasa a ser 6-10px (detectable)
        upscaled = cv2.resize(crop, None, fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_LINEAR)

        results = self.model_ball(upscaled, conf=0.08, verbose=False)[0]
        if not results.boxes:
            return []

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            # Escalar de vuelta a coordenadas originales y sumar el offset del crop
            detections.append((
                x1 / 2.0,
                y1 / 2.0 + roi_top,
                x2 / 2.0,
                y2 / 2.0 + roi_top,
                conf,
            ))

        return detections

    def draw_annotations(self, frame, results):
        return results.plot()
