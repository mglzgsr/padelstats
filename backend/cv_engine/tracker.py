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
    """

    # Límite vertical que separa las dos zonas (fracción de altura del frame)
    ZONE_BOUNDARY = 0.52

    def __init__(self, model_path='yolov8n.pt', ball_model_path='backend/tennis_ball_best.pt'):
        self.model_persons = YOLO(model_path)
        self.model_persons.to('mps')

        self.model_ball = YOLO(ball_model_path)
        self.model_ball.to('mps')

        # Estado de los 4 slots
        # Cada slot: {"last_pos": (x,y), "last_frame": int, "yolo_id": int, "hist": ndarray, "zone": str}
        self.slots = {1: None, 2: None, 3: None, 4: None}

        # Qué zona pertenece a cada slot
        self.slot_zones = {1: "near", 2: "near", 3: "far", 4: "far"}

        self.max_lost_frames = 300   # ~10 segundos a 30fps
        self.max_distance = 250      # píxeles máximos entre frames consecutivos

        # Peso del histograma nuevo en la media exponencial (EWA)
        # Bajo = memoria larga, conservador. Alto = adaptación rápida.
        self.hist_alpha = 0.25

        self.frame_height = None  # Se asigna al primer frame

    # ------------------------------------------------------------------
    # Utilidades privadas
    # ------------------------------------------------------------------

    def _get_zone(self, pos_y: float) -> str:
        """Clasifica una posición Y como 'near' (abajo) o 'far' (arriba)."""
        if self.frame_height is None:
            return "near"
        boundary = self.frame_height * self.ZONE_BOUNDARY
        return "near" if pos_y > boundary else "far"

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

    def _hist_distance(self, h1, h2) -> float:
        """Chi-cuadrado normalizado entre dos histogramas. 0 = idénticos, 1 = máximos."""
        if h1 is None or h2 is None:
            return 0.5  # Neutro si no hay datos
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
        return min(dist, 2.0) / 2.0

    # ------------------------------------------------------------------
    # Lógica de asignación
    # ------------------------------------------------------------------

    def _assign_zone(self, zone_detections: list, zone_slot_ids: list, frame_count: int):
        """
        Asigna detecciones a slots de una zona usando el algoritmo húngaro.
        Modifica self.slots in-place y retorna {yolo_id: slot_id}.
        """
        mapping = {}

        if not zone_detections:
            return mapping

        active_slots = [sid for sid in zone_slot_ids if self.slots[sid] is not None]
        empty_slots = [sid for sid in zone_slot_ids if self.slots[sid] is None]

        assigned_det_indices = set()

        # --- Fase 1: Asignación óptima a slots activos (algoritmo húngaro) ---
        if active_slots:
            n_dets = len(zone_detections)
            n_slots = len(active_slots)
            cost = np.full((n_dets, n_slots), fill_value=2.0)

            for r, d in enumerate(zone_detections):
                for c, sid in enumerate(active_slots):
                    data = self.slots[sid]
                    dist = np.hypot(
                        d["pos"][0] - data["last_pos"][0],
                        d["pos"][1] - data["last_pos"][1]
                    )
                    # Rechazar si demasiado lejos (hard constraint)
                    if dist > self.max_distance * 2.0:
                        cost[r, c] = 2.0
                        continue

                    spatial = min(dist / self.max_distance, 1.0)
                    color = self._hist_distance(d["hist"], data.get("hist"))
                    cost[r, c] = 0.55 * color + 0.45 * spatial

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 0.75:  # Umbral de aceptación
                    continue
                sid = active_slots[c]
                d = zone_detections[r]
                self.slots[sid] = {
                    "last_pos": d["pos"],
                    "last_frame": frame_count,
                    "yolo_id": d["id"],
                    "hist": self._blend_hist(self.slots[sid].get("hist"), d["hist"]),
                    "zone": d["zone"],
                }
                mapping[d["id"]] = sid
                assigned_det_indices.add(r)

        # --- Fase 2: Detecciones no asignadas llenan slots vacíos ---
        unassigned = [
            (r, d) for r, d in enumerate(zone_detections)
            if r not in assigned_det_indices
        ]
        # Ordenar por X para asignación consistente (izquierda primero)
        unassigned.sort(key=lambda rd: rd[1]["pos"][0])

        for (r, d), sid in zip(unassigned, empty_slots):
            self.slots[sid] = {
                "last_pos": d["pos"],
                "last_frame": frame_count,
                "yolo_id": d["id"],
                "hist": d["hist"],
                "zone": d["zone"],
            }
            mapping[d["id"]] = sid

        return mapping

    def _update_slots(self, detections: list, frame_count: int, frame=None) -> dict:
        """
        Actualiza los slots con las detecciones del frame actual.
        Retorna {yolo_id: slot_id}.
        """
        if self.frame_height is None and frame is not None:
            self.frame_height = frame.shape[0]

        # Calcular histogramas y zona para cada detección
        for d in detections:
            d["hist"] = self._get_color_histogram(frame, d["xyxy"]) if frame is not None else None
            d["zone"] = self._get_zone(d["pos"][1])

        # Separar por zona
        near_dets = [d for d in detections if d["zone"] == "near"]
        far_dets = [d for d in detections if d["zone"] == "far"]

        near_slots = [sid for sid, z in self.slot_zones.items() if z == "near"]
        far_slots = [sid for sid, z in self.slot_zones.items() if z == "far"]

        mapping = {}
        mapping.update(self._assign_zone(near_dets, near_slots, frame_count))
        mapping.update(self._assign_zone(far_dets, far_slots, frame_count))

        # Expirar slots sin actualización reciente
        for sid, data in self.slots.items():
            if data and (frame_count - data["last_frame"]) > self.max_lost_frames:
                self.slots[sid] = None

        return mapping

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def track_frame(self, frame, frame_count: int = 0) -> dict:
        # Detección de personas
        person_results = self.model_persons.track(
            frame, persist=True, classes=[0], conf=0.25, verbose=False
        )[0]

        mapping = {}
        if person_results.boxes and person_results.boxes.id is not None:
            detections = []
            for i, yolo_id in enumerate(person_results.boxes.id.cpu().numpy()):
                xyxy = person_results.boxes.xyxy[i].cpu().numpy()
                pos = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                detections.append({"id": int(yolo_id), "pos": pos, "xyxy": xyxy})

            mapping = self._update_slots(detections, frame_count, frame)

        person_results.slot_mapping = mapping

        # Detección de pelota
        ball_results = self.model_ball.track(frame, persist=True, conf=0.10, verbose=False)[0]

        return {
            "person_results": person_results,
            "ball_results": ball_results,
        }

    def draw_annotations(self, frame, results):
        return results.plot()
