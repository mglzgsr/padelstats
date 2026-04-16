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

    def __init__(self, fps=30.0):
        self.fps = fps
        self.shot_cooldown = max(15, int(fps * 0.5))
        self.last_shot_frame = {1: -999, 2: -999, 3: -999, 4: -999, 0: -999}

        # Buffer por jugador: [(dist, threshold, (bx, by), frame_idx), ...]
        self.player_buffers = {1: [], 2: [], 3: [], 4: []}

        # Ball Tracker con Kalman Filter para interpolación
        self.ball_tracker = BallTracker(fps=int(fps))
        print(f"[ShotClassifier] Inicializado con buffers por jugador (fps={fps})")

    # ------------------------------------------------------------------
    def detect_impact(self, frame_idx, ball_pos, players):
        """Detecta golpe con approach→min→departure por jugador."""
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
            p_height  = max(py2 - py1, 1)
            p_width   = max(px2 - px1, 1)
            # Distancia al borde más cercano del bbox (no al centro)
            # — más realista: la raqueta puede estar en cualquier punto del bbox
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            dist = np.sqrt((bx - cx)**2 + (by - cy)**2)

            # Threshold basado en altura del bbox
            threshold = max(80, min(p_height * 0.50, 200))

            buf = self.player_buffers[pid]
            buf.append((dist, threshold, (bx, by), frame_idx))
            if len(buf) > self.BUFFER_SIZE:
                buf.pop(0)

        # Debug cada 300 frames: mostrar distancias actuales a cada jugador
        if frame_idx % 300 == 0:
            for pid, buf in self.player_buffers.items():
                if buf:
                    d, thr, _, _ = buf[-1]
                    print(f"[ShotDebug] Frame {frame_idx} J{pid}: dist={d:.0f}px threshold={thr:.0f}px")

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

            # El mínimo debe estar "en el interior" de la ventana
            margin = 2
            if min_idx < margin or min_idx > len(recent) - margin - 1:
                continue

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
                # Debug si estamos cerca del threshold
                if min_dist < min_thr * 1.5 and frame_idx % 100 == 0:
                    print(f"[ShotDebug] J{pid} frame={min_frame} dist={min_dist:.0f}/{min_thr:.0f} "
                          f"app={'OK' if is_app else 'FAIL'} dep={'OK' if is_dep else 'FAIL'} "
                          f"app({app_first:.0f}→{app_second:.0f}) dep({dep_first:.0f}→{dep_second:.0f})")
                continue

            # Cooldown por jugador
            if min_frame - self.last_shot_frame.get(pid, -999) < self.shot_cooldown:
                continue

            # Elegir el mejor candidato (menor distancia al mínimo)
            if min_dist < best_dist:
                best_dist   = min_dist
                best_impact = {"frame": min_frame, "pos": min_pos,
                               "player_id": pid, "min_dist": min_dist,
                               "min_thr": min_thr}

        if best_impact:
            pid = best_impact["player_id"]
            print(f"[Shot] Frame {frame_idx}: GOLPE J{pid} "
                  f"(dist={best_impact['min_dist']:.1f}px, thr={best_impact['min_thr']:.1f}px)")
            self.last_shot_frame[pid] = best_impact["frame"]
            # Clasificar tipo: smash/bandeja si pelota está en la parte alta del bbox
            # Necesitamos la bbox del jugador en el momento del impacto
            return best_impact

        return None

    # ------------------------------------------------------------------
    def classify_shot(self, impact, players):
        if not impact:
            return None
        pid  = impact["player_id"]
        pos  = impact["pos"]
        bx, by = pos

        # Buscar bbox del jugador para clasificar
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
            "pos":        pos,
            "event":      "Shot",
        }
