import numpy as np
from .ball_tracker import BallTracker

class ShotClassifier:
    """Detector de golpes basado en proximidad pelota-jugador."""

    BUFFER_SIZE    = 40   # historial de distancias por jugador
    RECENT_WINDOW  = 14   # ventana de análisis (frames)
    MIN_APPROACH   = 2    # puntos mínimos antes del mínimo
    MIN_DEPARTURE  = 2    # puntos mínimos después del mínimo
    APP_RATIO      = 0.92 # segunda mitad del approach < primera mitad * ratio
    DEP_RATIO      = 1.08 # segunda mitad del departure > primera mitad * ratio
    MIN_DIR_CHANGE = 30   # grados mínimos de cambio de dirección (fly-by < 30°, golpe real > 30°)
    GLOBAL_COOLDOWN = 20  # frames de bloqueo global después de cualquier golpe (~0.67s a 30fps)

    def __init__(self, fps=30.0):
        self.fps = fps
        self.shot_cooldown = max(15, int(fps * 0.5))
        self.last_shot_frame = {1: -999, 2: -999, 3: -999, 4: -999, 0: -999}
        self.last_any_shot_frame = -999  # cooldown global (sólo 1 golpe a la vez)

        # Buffer por jugador: [(dist, threshold, (bx, by), frame_idx), ...]
        self.player_buffers = {1: [], 2: [], 3: [], 4: []}

        # Ball Tracker con Kalman Filter para interpolación
        self.ball_tracker = BallTracker(fps=int(fps))
        print(f"[ShotClassifier] Inicializado con buffers por jugador (fps={fps})")

    # ------------------------------------------------------------------
    def detect_impact(self, frame_idx, ball_pos, players):
        """Detecta golpe con approach→min→departure + check de cambio de dirección."""
        if not players:
            return None

        # Actualizar ball tracker (interpola si ball_pos es None)
        if ball_pos is not None:
            bx, by = ball_pos
            ball_tracked = self.ball_tracker.update(frame_idx, (bx, by, 0.5))
        else:
            ball_tracked = self.ball_tracker.update(frame_idx, None)

        if ball_tracked is None:
            return None

        bx, by, conf = ball_tracked

        # --- Actualizar buffers por jugador ---
        for p in players:
            px1, py1, px2, py2, pid = p
            if pid <= 0:
                continue
            if pid not in self.player_buffers:
                continue
            p_height = max(py2 - py1, 1)
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            dist = np.sqrt((bx - cx)**2 + (by - cy)**2)

            # Threshold: pelota debe estar dentro o muy cerca del bbox del jugador.
            # Para un jugador de 200px de alto → max(50, 80, 100) = 80px del centro.
            threshold = max(50, min(p_height * 0.40, 100))

            buf = self.player_buffers[pid]
            buf.append((dist, threshold, (bx, by), frame_idx))
            if len(buf) > self.BUFFER_SIZE:
                buf.pop(0)

        # Debug cada 300 frames
        if frame_idx % 300 == 0:
            for pid, buf in self.player_buffers.items():
                if buf:
                    d, thr, _, _ = buf[-1]
                    print(f"[ShotDebug] Frame {frame_idx} J{pid}: dist={d:.0f}px thr={thr:.0f}px")

        # Bloqueo global: si acaba de detectarse un golpe, esperar antes de detectar otro
        if frame_idx - self.last_any_shot_frame < self.GLOBAL_COOLDOWN:
            return None

        # --- Comprobar patrón approach→min→departure por jugador ---
        best_impact = None
        best_dist   = float('inf')

        for pid, buf in self.player_buffers.items():
            if len(buf) < 5:
                continue

            recent = buf[-self.RECENT_WINDOW:]
            if len(recent) < 5:
                continue

            min_idx = min(range(len(recent)), key=lambda i: recent[i][0])
            min_dist, min_thr, min_pos, min_frame = recent[min_idx]

            # El mínimo debe estar en el interior de la ventana (margen de 2)
            margin = 2
            if min_idx < margin or min_idx > len(recent) - margin - 1:
                continue

            # Pelota debe estar dentro o muy cerca del bbox
            if min_dist > min_thr:
                continue

            approach  = [d for d, _, _, _ in recent[:min_idx][-4:]]
            departure = [d for d, _, _, _ in recent[min_idx + 1:][:4]]

            if len(approach) < self.MIN_APPROACH or len(departure) < self.MIN_DEPARTURE:
                continue

            app_first  = np.mean(approach[:len(approach)//2 + 1])
            app_second = np.mean(approach[len(approach)//2:])
            dep_first  = np.mean(departure[:len(departure)//2 + 1])
            dep_second = np.mean(departure[len(departure)//2:])

            is_app = app_second < app_first * self.APP_RATIO
            is_dep = dep_second > dep_first * self.DEP_RATIO

            if not (is_app and is_dep):
                if min_dist < min_thr * 1.5 and frame_idx % 150 == 0:
                    print(f"[ShotDebug] J{pid} f={min_frame} dist={min_dist:.0f}/{min_thr:.0f} "
                          f"app={'OK' if is_app else 'FAIL'}({app_first:.0f}→{app_second:.0f}) "
                          f"dep={'OK' if is_dep else 'FAIL'}({dep_first:.0f}→{dep_second:.0f})")
                continue

            # --- Check cambio de dirección ---
            # Un fly-by mantiene la dirección (<30°); un golpe real la cambia (>30°)
            before_pos = [entry[2] for entry in recent[:min_idx][-3:]]
            after_pos  = [entry[2] for entry in recent[min_idx + 1:][:3]]

            dir_ok = True  # asumir OK si no hay suficientes puntos para calcular
            if len(before_pos) >= 2 and len(after_pos) >= 2:
                vb = (before_pos[-1][0] - before_pos[-2][0],
                      before_pos[-1][1] - before_pos[-2][1])
                va = (after_pos[1][0]  - after_pos[0][0],
                      after_pos[1][1]  - after_pos[0][1])
                mag_b = (vb[0]**2 + vb[1]**2) ** 0.5
                mag_a = (va[0]**2 + va[1]**2) ** 0.5
                if mag_b > 2 and mag_a > 2:
                    cos_a = np.clip((vb[0]*va[0] + vb[1]*va[1]) / (mag_b * mag_a), -1, 1)
                    angle_deg = float(np.degrees(np.arccos(cos_a)))
                    dir_ok = angle_deg >= self.MIN_DIR_CHANGE
                    if not dir_ok and min_dist < min_thr * 1.5:
                        print(f"[ShotDebug] J{pid} f={min_frame} RECHAZADO fly-by "
                              f"(ángulo={angle_deg:.1f}° < {self.MIN_DIR_CHANGE}°)")

            if not dir_ok:
                continue

            # Cooldown por jugador
            if min_frame - self.last_shot_frame.get(pid, -999) < self.shot_cooldown:
                continue

            # Elegir el mejor candidato (menor distancia al mínimo)
            if min_dist < best_dist:
                best_dist   = min_dist
                best_impact = {
                    "frame":     min_frame,
                    "pos":       min_pos,
                    "player_id": pid,
                    "min_dist":  min_dist,
                    "min_thr":   min_thr,
                }

        if best_impact:
            pid = best_impact["player_id"]
            print(f"[Shot] Frame {frame_idx}: GOLPE J{pid} "
                  f"(dist={best_impact['min_dist']:.1f}px thr={best_impact['min_thr']:.1f}px)")
            # Cooldown individual + global
            self.last_shot_frame[pid] = best_impact["frame"]
            self.last_any_shot_frame  = best_impact["frame"]
            return best_impact

        return None

    # ------------------------------------------------------------------
    def classify_shot(self, impact, players):
        if not impact:
            return None
        pid    = impact["player_id"]
        bx, by = impact["pos"]

        shot_type = "Stroke"
        for px1, py1, px2, py2, p_id in players:
            if p_id == pid:
                p_height = py2 - py1
                if by < py1 + p_height * 0.35:
                    shot_type = "Smash/Bandeja"
                break

        return {
            "frame":      impact["frame"],
            "player_id":  pid,
            "shot_type":  shot_type,
            "pos":        impact["pos"],
            "event":      "Shot",
        }
