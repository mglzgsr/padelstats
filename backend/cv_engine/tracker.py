from __future__ import annotations

import os
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import supervision as sv
from .ball_tracker import BallTracker

# Dispositivo de inferencia: cuda (GPU NVIDIA) > mps (Apple Silicon) > cpu
_DEVICE = os.environ.get("TORCH_DEVICE", "")
if not _DEVICE:
    import torch
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        _DEVICE = "mps"
    else:
        _DEVICE = "cpu"


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

    # Buffer alrededor de la red (fracción de altura).
    # Un jugador dentro del buffer se considera "zona ambigua": no se penaliza
    # por cruzar zona, evitando swaps cuando se acerca a la red.
    NET_Y_BUFFER = 0.05  # ±54px en 1080p

    def __init__(self, model_path='yolov8l.pt', ball_model_path='backend/tennis_ball_roboflow_v2.pt', net_y_fraction=None):
        print(f"[Tracker] Cargando modelo YOLO: {model_path}")
        self.model_persons = YOLO(model_path)
        self.model_persons.to(_DEVICE)
        print(f"[Tracker] Modelo cargado - Parámetros: {sum(p.numel() for p in self.model_persons.model.parameters())/1e6:.1f}M")

        if ball_model_path and os.path.isfile(ball_model_path):
            print(f"[Tracker] Cargando modelo BALL: {ball_model_path}")
            self.model_ball = YOLO(ball_model_path)
            self.model_ball.to(_DEVICE)
            print(f"[Tracker] Modelo BALL cargado - Parámetros: {sum(p.numel() for p in self.model_ball.model.parameters())/1e6:.1f}M")
        else:
            print(f"[Tracker] Modelo BALL no encontrado ({ball_model_path}) — detección de pelota delegada a TrackNetV3")
            self.model_ball = None

        # Posición de la red como fracción de la altura del frame.
        # Si se proporciona net_y_fraction (de court_config.json), se usa ese valor.
        # De lo contrario, se usa un valor por defecto (típico cámara fondo: 0.60-0.68).
        self.NET_Y = net_y_fraction if net_y_fraction is not None else 0.63
        print(f"[Tracker] Dispositivo de inferencia: {_DEVICE}")
        print(f"[Tracker] NET_Y configurado en {self.NET_Y:.3f}")

        # ========== Supervision ByteTrack (como padel_analytics) ==========
        print(f"[Tracker] Inicializando Supervision ByteTrack...")
        # IMPORTANTE: Inicializar después porque necesita fps del video
        self.byte_tracker = None  # Se inicializará en el primer frame
        print(f"[Tracker] Supervision ByteTrack - usando defaults como padel_analytics")

        # ========== SISTEMA DE SLOTS (asignación estable J1-J4) ==========
        self.initialized = False
        self.initialization_buffer = []
        self.INIT_FRAMES = 30

        # yolo_to_player se mantiene solo para la inicialización inicial
        self.yolo_to_player = {}  # {yolo_id: slot}

        # Estado para _update_slots (Hungarian + color histograms)
        self.slots = {1: None, 2: None, 3: None, 4: None}
        self.slot_zones = {1: 'near', 2: 'near', 3: 'far', 4: 'far'}
        self.max_distance = 400       # px – umbral espacial para matching
        self.max_lost_frames = 300    # frames antes de expirar un slot (~10s a 30fps)
        self.CONFIRM_FRAMES = 3       # frames consecutivos para confirmar reasignación
        self._pending = {}            # {yolo_id: (slot, count)}
        self.hist_alpha = 0.2         # peso del histograma nuevo en el blend EWA
        self.ZONE_FLIP_FRAMES = 15    # frames consecutivos para confirmar cambio de zona
        self._zone_wrong_frames = {1: 0, 2: 0, 3: 0, 4: 0}

        # ========== Ball Tracker con Kalman Filter e interpolación ==========
        self.ball_tracker = None  # Se inicializará con fps del video
        print(f"[Tracker] Ball Tracker con interpolación: Pendiente (se inicializa con fps)")

        # Mantener compatibilidad con código antiguo que usa POSITION_THRESHOLD
        self.last_position = {1: None, 2: None, 3: None, 4: None}
        self.last_seen_frame = {1: -999, 2: -999, 3: -999, 4: -999}
        self.POSITION_THRESHOLD = 400
        self.MAX_MISSING_FRAMES = 300

        self.frame_height = None  # Se asigna al primer frame

    # ------------------------------------------------------------------
    # Inicialización de jugadores (primeros frames)
    # ------------------------------------------------------------------

    def _initialize_players(self, detections, frame):
        """
        Asigna J1-J4 basándose en posición inicial de los jugadores.
        Se ejecuta una sola vez al inicio del video.

        Lógica de asignación FLEXIBLE:
        - Ordena 4 jugadores por posición Y (pies)
        - 2 más alejados (Y menor) → J3, J4 (zona far)
        - 2 más cercanos (Y mayor) → J1, J2 (zona near)
        - Dentro de cada pareja, ordena por X (izq/der)
        """
        try:
            print(f"[Tracker INIT] ===== INICIANDO INICIALIZACIÓN =====")
            print(f"[Tracker INIT] Número de detecciones recibidas: {len(detections)}")

            if len(detections) != 4:
                print(f"[Tracker INIT DEBUG] ❌ Solo {len(detections)} jugadores - necesitamos 4")
                return False  # Necesitamos exactamente 4 jugadores para inicializar

            # Extraer info de jugadores
            print(f"[Tracker INIT] Extrayendo información de jugadores...")
            players = []
            for i, det in enumerate(detections):
                yolo_id = det["id"]
                xyxy = det["xyxy"]
                feet_y = xyxy[3]  # Coordenada Y de los pies (mayor = más cerca de cámara)
                center_x = (xyxy[0] + xyxy[2]) / 2
                players.append({"yolo_id": yolo_id, "x": center_x, "y": feet_y})
                print(f"[Tracker INIT]   Detección {i}: YOLO_ID={yolo_id}, X={center_x:.1f}, Y={feet_y:.1f}")

            # Ordenar por Y (de menor a mayor = de fondo a cerca)
            print(f"[Tracker INIT] Ordenando por Y (menor=lejos, mayor=cerca)...")
            players.sort(key=lambda p: p["y"])
            for i, p in enumerate(players):
                print(f"[Tracker INIT]   Posición {i}: YOLO_ID={p['yolo_id']}, Y={p['y']:.1f}")

            # Los 2 primeros (Y menor) son zona FAR
            # Los 2 últimos (Y mayor) son zona NEAR
            far_players = players[:2]
            near_players = players[2:]
            print(f"[Tracker INIT] Dividiendo en zonas:")
            print(f"[Tracker INIT]   FAR:  IDs {[p['yolo_id'] for p in far_players]}")
            print(f"[Tracker INIT]   NEAR: IDs {[p['yolo_id'] for p in near_players]}")

            # Ordenar cada grupo por X (izquierda a derecha)
            print(f"[Tracker INIT] Ordenando cada zona por X (izq→der)...")
            far_players.sort(key=lambda p: p["x"])
            near_players.sort(key=lambda p: p["x"])
            print(f"[Tracker INIT]   FAR ordenado:  IDs {[p['yolo_id'] for p in far_players]}")
            print(f"[Tracker INIT]   NEAR ordenado: IDs {[p['yolo_id'] for p in near_players]}")

            # Asignar slots
            print(f"[Tracker INIT] Asignando slots...")
            self.yolo_to_player[near_players[0]["yolo_id"]] = 1  # J1: near-izq
            print(f"[Tracker INIT]   J1 (near-izq) = YOLO_ID {near_players[0]['yolo_id']}")

            self.yolo_to_player[near_players[1]["yolo_id"]] = 2  # J2: near-der
            print(f"[Tracker INIT]   J2 (near-der) = YOLO_ID {near_players[1]['yolo_id']}")

            self.yolo_to_player[far_players[0]["yolo_id"]] = 3   # J3: far-izq
            print(f"[Tracker INIT]   J3 (far-izq)  = YOLO_ID {far_players[0]['yolo_id']}")

            self.yolo_to_player[far_players[1]["yolo_id"]] = 4   # J4: far-der
            print(f"[Tracker INIT]   J4 (far-der)  = YOLO_ID {far_players[1]['yolo_id']}")

            print(f"[Tracker] ✅ Jugadores inicializados:")
            print(f"  J1 (near-izq): YOLO ID {near_players[0]['yolo_id']} @ Y={near_players[0]['y']:.0f}, X={near_players[0]['x']:.0f}")
            print(f"  J2 (near-der): YOLO ID {near_players[1]['yolo_id']} @ Y={near_players[1]['y']:.0f}, X={near_players[1]['x']:.0f}")
            print(f"  J3 (far-izq):  YOLO ID {far_players[0]['yolo_id']} @ Y={far_players[0]['y']:.0f}, X={far_players[0]['x']:.0f}")
            print(f"  J4 (far-der):  YOLO ID {far_players[1]['yolo_id']} @ Y={far_players[1]['y']:.0f}, X={far_players[1]['x']:.0f}")
            print(f"[Tracker INIT] Diccionario yolo_to_player: {self.yolo_to_player}")

            # Inicializar self.slots con histogramas reales desde el primer frame
            # Esto es crítico: sin histogramas iniciales, _update_slots solo usa
            # distancia espacial y falla cuando los jugadores se cruzan.
            det_by_id = {d["id"]: d for d in detections}
            for yolo_id, slot in self.yolo_to_player.items():
                det = det_by_id.get(yolo_id)
                if det:
                    initial_hist = self._get_color_histogram(frame, det["xyxy"]) if frame is not None else None
                    self.slots[slot] = {
                        "last_pos":   det["pos"],
                        "last_frame": 0,
                        "yolo_id":    yolo_id,
                        "hist":       initial_hist,
                        "zone":       self.slot_zones[slot],
                    }
            print(f"[Tracker INIT] Histogramas iniciales: { {s: 'OK' if v and v.get('hist') is not None else 'None' for s,v in self.slots.items()} }")

            self.initialized = True
            print(f"[Tracker INIT] ===== INICIALIZACIÓN COMPLETADA =====")
            return True

        except Exception as e:
            print(f"[Tracker INIT ERROR] ❌ Excepción durante inicialización:")
            print(f"[Tracker INIT ERROR]   Tipo: {type(e).__name__}")
            print(f"[Tracker INIT ERROR]   Mensaje: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _reassign_missing_players(self, detections_list, frame_count):
        """
        Sistema de memoria temporal con re-asignación inteligente de IDs.

        Lógica:
        1. Actualizar última posición de jugadores conocidos que siguen presentes
        2. Detectar qué slots (J1-J4) están MISSING en este frame
        3. Detectar IDs nuevos (no asignados a ningún slot)
        4. Para cada ID nuevo, buscar el slot missing más cercano por posición
        5. Si la distancia es razonable, re-asignar ese ID al slot

        Esto permite que si J1 sale del encuadre (pierde ID=5) y vuelve con ID=45,
        el sistema detecte que:
        - Falta J1
        - ID=45 está cerca de la última posición de J1
        - Re-asigna: J1 → ID=45
        """
        if frame_count % 100 == 0:
            print(f"[Tracker ReID DEBUG] Frame {frame_count}: _reassign_missing_players llamado con {len(detections_list)} detecciones")

        # 1. Actualizar posiciones de jugadores que siguen presentes
        current_yolo_ids = {det["id"] for det in detections_list}

        for yolo_id, slot in self.yolo_to_player.items():
            if yolo_id in current_yolo_ids:
                # Este jugador sigue aquí, actualizar su posición
                det = next(d for d in detections_list if d["id"] == yolo_id)
                xyxy = det["xyxy"]
                feet_pos = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
                self.last_position[slot] = feet_pos
                self.last_seen_frame[slot] = frame_count

        # 2. Identificar slots MISSING (jugadores que deberían estar pero no se detectan)
        missing_slots = []
        for yolo_id, slot in self.yolo_to_player.items():
            if yolo_id not in current_yolo_ids:
                # Este jugador no está en las detecciones actuales
                frames_missing = frame_count - self.last_seen_frame.get(slot, frame_count)
                if frames_missing <= self.MAX_MISSING_FRAMES:
                    missing_slots.append(slot)

        # 3. Identificar IDs nuevos (no asignados a ningún slot)
        assigned_yolo_ids = set(self.yolo_to_player.values())
        new_yolo_ids = current_yolo_ids - assigned_yolo_ids

        if frame_count % 300 == 0 and (new_yolo_ids or missing_slots):
            print(f"[Tracker ReID DEBUG] Frame {frame_count}: missing_slots={missing_slots}, new_yolo_ids={new_yolo_ids}")
            print(f"[Tracker ReID DEBUG] yolo_to_player={self.yolo_to_player}")

        if not new_yolo_ids or not missing_slots:
            return  # No hay nada que re-asignar

        # 4. Intentar re-asignar IDs nuevos a slots missing por proximidad
        reassignments = []
        for new_id in list(new_yolo_ids):
            det = next(d for d in detections_list if d["id"] == new_id)
            xyxy = det["xyxy"]
            new_feet_pos = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])

            # Buscar el slot missing más cercano a este nuevo ID
            best_slot = None
            best_distance = float('inf')

            for slot in missing_slots:
                if self.last_position[slot] is None:
                    continue  # Sin posición conocida, no podemos calcular distancia

                last_pos = self.last_position[slot]
                distance = np.sqrt(
                    (new_feet_pos[0] - last_pos[0])**2 +
                    (new_feet_pos[1] - last_pos[1])**2
                )

                if distance < best_distance and distance < self.POSITION_THRESHOLD:
                    best_distance = distance
                    best_slot = slot

            # Fallback: Si no encontró por proximidad, asignar por zona de cancha
            if best_slot is None and missing_slots:
                # Determinar zona del nuevo ID (near/far, left/right)
                new_y = new_feet_pos[1]
                new_x = new_feet_pos[0]
                net_y = self.frame_height * self.NET_Y if self.frame_height else 9999
                frame_center_x = (self.frame_height * 2) if self.frame_height else 1920  # approx width

                is_near = new_y > net_y
                is_left = new_x < frame_center_x

                if frame_count % 300 == 0:
                    print(f"[Tracker ReID DEBUG] Fallback para ID {new_id}: pos=({new_x:.0f}, {new_y:.0f}), near={is_near}, left={is_left}, missing={missing_slots}")

                # Mapeo de zona a slot: J1=near-left, J2=near-right, J3=far-left, J4=far-right
                # Intentar asignar a la zona exacta primero
                if is_near and is_left and 1 in missing_slots:
                    best_slot = 1
                    best_distance = 999  # Marca como fallback
                elif is_near and not is_left and 2 in missing_slots:
                    best_slot = 2
                    best_distance = 999
                elif not is_near and is_left and 3 in missing_slots:
                    best_slot = 3
                    best_distance = 999
                elif not is_near and not is_left and 4 in missing_slots:
                    best_slot = 4
                    best_distance = 999
                else:
                    # Fallback más flexible: si no puede asignar a la zona exacta,
                    # asignar al primer slot disponible (mejor que nada)
                    if missing_slots:
                        best_slot = missing_slots[0]
                        best_distance = 999
                        if frame_count % 300 == 0:
                            print(f"[Tracker ReID DEBUG] Asignación flexible: ID {new_id} → slot {best_slot} (no match perfecto)")

            if best_slot is not None:
                # Re-asignar este nuevo ID al slot faltante
                # yolo_to_player es {yolo_id: slot} — buscar el yolo_id antiguo del slot
                old_yolo_id = next((yid for yid, s in self.yolo_to_player.items() if s == best_slot), None)
                if old_yolo_id is not None:
                    del self.yolo_to_player[old_yolo_id]
                self.yolo_to_player[new_id] = best_slot
                reassignments.append((best_slot, old_yolo_id, new_id, best_distance))

                # Actualizar posición y frame
                self.last_position[best_slot] = new_feet_pos
                self.last_seen_frame[best_slot] = frame_count

                # Remover de las listas para no re-asignar múltiples veces
                new_yolo_ids.remove(new_id)
                missing_slots.remove(best_slot)

        # Log de re-asignaciones
        if reassignments:
            print(f"[Tracker ReID] Frame {frame_count}: RE-ASIGNACIONES detectadas:")
            for slot, old_id, new_id, dist in reassignments:
                print(f"  J{slot}: YOLO_ID {old_id} → {new_id} (distancia={dist:.1f}px)")

    # ------------------------------------------------------------------
    # Utilidades privadas (deprecadas - mantener por compatibilidad)
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
                yid = d["id"]

                # Confirmation buffer: si el slot que se propone es diferente al
                # que este yolo_id tenía antes, exigir CONFIRM_FRAMES consecutivos
                # antes de aplicar el cambio para evitar flicker por frames ruidosos.
                current_sid = next(
                    (s for s, data in self.slots.items()
                     if data and data.get("yolo_id") == yid),
                    None
                )
                if current_sid is not None and current_sid != sid:
                    pend = self._pending.get(yid)
                    if pend and pend[0] == sid:
                        count = pend[1] + 1
                    else:
                        count = 1
                    self._pending[yid] = (sid, count)
                    if count < self.CONFIRM_FRAMES:
                        # Aún no confirmado: mantener slot actual y actualizar posición
                        self._maybe_flip_zone(current_sid, d["xyxy"][3])
                        self.slots[current_sid]["last_pos"]   = d["pos"]
                        self.slots[current_sid]["last_frame"] = frame_count
                        self.slots[current_sid]["hist"] = self._blend_hist(
                            self.slots[current_sid].get("hist"), d["hist"]
                        )
                        mapping[yid] = current_sid
                        assigned_det_indices.add(r)
                        continue
                    # Confirmado: aplicar el nuevo slot
                    del self._pending[yid]
                else:
                    self._pending.pop(yid, None)

                # Actualizar zona adaptativa antes de guardar en el slot
                self._maybe_flip_zone(sid, d["xyxy"][3])
                self.slots[sid] = {
                    "last_pos":   d["pos"],
                    "last_frame": frame_count,
                    "yolo_id":    yid,
                    "hist":       self._blend_hist(self.slots[sid].get("hist") if self.slots[sid] else None, d["hist"]),
                    "zone":       self.slot_zones[sid],
                }
                mapping[yid] = sid
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

        # Imponer orden lateral dentro de cada par de zona.
        # Si J1 está a la derecha de J2 (o J3 a la derecha de J4), se han cruzado
        # en el mapping: intercambiar slots y corregir el mapping resultante.
        for s_left, s_right in [(1, 2), (3, 4)]:
            if self.slots[s_left] is not None and self.slots[s_right] is not None:
                x_left  = self.slots[s_left]["last_pos"][0]
                x_right = self.slots[s_right]["last_pos"][0]
                if x_left > x_right + 40:  # Cruce claro (>40 px)
                    self.slots[s_left], self.slots[s_right] = (
                        self.slots[s_right], self.slots[s_left]
                    )
                    for yolo_id_k, sid in list(mapping.items()):
                        if sid == s_left:
                            mapping[yolo_id_k] = s_right
                        elif sid == s_right:
                            mapping[yolo_id_k] = s_left

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
        # Inicializar frame_height en el primer frame
        if self.frame_height is None:
            self.frame_height = frame.shape[0]

        # ========== Supervision ByteTrack (exacto como padel_analytics) ==========
        # Inicializar ByteTrack en el primer frame con el fps del video
        if self.byte_tracker is None and frame_count == 0:
            # Calcular FPS del video (30 por defecto si no se puede determinar)
            video_fps = 30.0
            self.byte_tracker = sv.ByteTrack(frame_rate=video_fps)
            print(f"[Tracker] Supervision ByteTrack inicializado (fps={video_fps})")

            # Inicializar Ball Tracker con interpolación
            self.ball_tracker = BallTracker(fps=int(video_fps))
            print(f"[Tracker] Ball Tracker con Kalman Filter inicializado (fps={video_fps})")

        # 1. Detección YOLO (sin tracking)
        person_results = self.model_persons.predict(
            frame, classes=[0], conf=0.3, verbose=False  # Reducido de 0.5 a 0.3
        )[0]

        # 2. Convertir a formato Supervision
        detections = sv.Detections.from_ultralytics(person_results)

        # 3. Filtrar por polígono de cancha
        # Permitir 80px de margen fuera del polígono: jugadores contra el cristal
        # tienen los pies justo en el borde o ligeramente fuera.
        PLAYER_POLYGON_MARGIN = -80  # px (negativo = fuera del polígono)
        if court_polygon is not None:
            before_filter = len(detections)
            mask = []
            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                feet = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
                dist = cv2.pointPolygonTest(court_polygon, feet, True)  # measureDist=True
                mask.append(dist >= PLAYER_POLYGON_MARGIN)
            detections = detections[np.array(mask)]
            after_filter = len(detections)

            # Log cada 100 frames para debug
            if frame_count % 100 == 0:
                print(f"[Tracker DEBUG] Frame {frame_count}: Detectados={before_filter}, Después filtro polígono={after_filter}")

        # 4. Actualizar ByteTrack con detecciones
        detections = self.byte_tracker.update_with_detections(detections=detections)

        if frame_count == 0:
            print(f"[Tracker] Supervision ByteTrack activado")
            print(f"[Tracker]   Detecciones tracked: {len(detections)}")

        # ========== Procesar detections de Supervision ==========
        mapping = {}
        detections_list = []
        if len(detections) > 0:
            # Limitar a 4 jugadores máximo
            if len(detections) > 4:
                # Ordenar por confianza y tomar top 4
                conf_indices = np.argsort(detections.confidence)[::-1][:4]
                detections = detections[conf_indices]

            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
                conf = detections.confidence[i] if detections.confidence is not None else 0.0

                if track_id is None:
                    continue

                pos = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                detections_list.append({
                    "id": int(track_id),
                    "pos": pos,
                    "xyxy": xyxy,
                    "conf": float(conf)
                })

            # ========== NUEVO SISTEMA: Inicialización + Mapeo permanente ==========
            if not self.initialized:
                # Fase de inicialización: acumular frames con 4 jugadores
                if frame_count % 100 == 0 and frame_count < 200:
                    print(f"[Tracker DEBUG] Frame {frame_count}: {len(detections_list)} jugadores detectados")

                if len(detections_list) == 4:
                    self.initialization_buffer.append((detections_list, frame))

                    # Intentar inicializar después de INIT_FRAMES frames
                    if len(self.initialization_buffer) >= self.INIT_FRAMES:
                        # Usar el frame más reciente para inicialización
                        recent_detections, recent_frame = self.initialization_buffer[-1]
                        print(f"[Tracker] Intentando inicializar con {len(recent_detections)} jugadores...")
                        if self._initialize_players(recent_detections, recent_frame):
                            print(f"[Tracker] Inicialización completada en frame {frame_count}")
                        else:
                            print(f"[Tracker] Inicialización fallida - reiniciar buffer")
                            self.initialization_buffer = []  # Reiniciar si falla

                # Pre-inicialización: todos los slots a 0 hasta tener 4 jugadores
                mapping = {det["id"]: 0 for det in detections_list}
            else:
                # Ya inicializado: usar _update_slots
                # Hungarian algorithm + color histograms → asignación estable sin depender de ByteTrack IDs
                mapping = self._update_slots(detections_list, frame_count, frame)

                # Log cada 300 frames
                if frame_count % 300 == 0:
                    active = {sid: (data["yolo_id"], data["last_pos"]) for sid, data in self.slots.items() if data}
                    print(f"[Tracker] Frame {frame_count} — slots activos: { {f'J{s}': f'id={v[0]} pos=({v[1][0]:.0f},{v[1][1]:.0f})' for s,v in active.items()} }")

        person_results.slot_mapping = mapping

        # Guardar las detecciones filtradas para que processor.py las use
        person_results.filtered_detections = detections

        # --- Detección de pelota con frame skipping ---
        if run_ball and self.model_ball is not None:
            # Usar .predict() en lugar de .track() para evitar que ByteTrack
            # bloquee nuevas detecciones después de perder el track inicial
            ball_results = self.model_ball.predict(
                frame, conf=0.05, verbose=False
            )[0]
        else:
            ball_results = None

        if run_far_zone and run_ball and self.model_ball is not None:
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

        # Ampliar 3× → la pelota de 3-5px pasa a ser 9-15px (muy detectable)
        upscaled = cv2.resize(crop, None, fx=3.0, fy=3.0,
                              interpolation=cv2.INTER_LINEAR)

        if self.model_ball is None:
            return []
        results = self.model_ball(upscaled, conf=0.05, verbose=False)[0]
        if not results.boxes:
            return []

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            # Escalar de vuelta a coordenadas originales y sumar el offset del crop
            detections.append((
                x1 / 3.0,
                y1 / 3.0 + roi_top,
                x2 / 3.0,
                y2 / 3.0 + roi_top,
                conf,
            ))

        return detections

    def draw_annotations(self, frame, results):
        return results.plot()
