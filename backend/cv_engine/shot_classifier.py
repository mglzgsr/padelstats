import numpy as np

class ShotClassifier:
    """
    Detector de golpes basado en cambio de dirección de la pelota.

    Con TrackNet al 99.9% tenemos la trayectoria casi completa de la pelota.
    Un golpe real SIEMPRE cambia la dirección del vuelo (>80°).
    Un fly-by o un pase cerca de un jugador NO cambia la dirección.

    Flujo:
    1. Acumular posiciones raw de la pelota (sin Kalman — más fiel al impacto real).
    2. Calcular vectores de velocidad antes y después de cada punto.
    3. Si el ángulo entre ambos vectores > MIN_ANGLE → posible golpe.
    4. El golpe se atribuye al jugador más cercano a la posición de impacto,
       siempre que esté a menos de MAX_PLAYER_DIST (filtra rebotes en cristal/red).
    5. Cooldown global de COOLDOWN_FRAMES para evitar dobles detecciones.
    """

    MIN_ANGLE        = 80    # grados — golpe real >80°, fly-by <30°
    MAX_PLAYER_DIST  = 500   # px — si no hay jugador cerca, es rebote en pared
    COOLDOWN_FRAMES  = 20    # frames de bloqueo global tras cada golpe (~0.67s a 30fps)
    HISTORY_SIZE     = 20    # frames de historial de trayectoria
    CHECK_OFFSET     = 5     # frames hacia atrás para buscar el punto de impacto
    VELOCITY_WINDOW  = 3     # frames para promediar velocidad antes/después

    def __init__(self, fps=30.0):
        self.fps = fps
        self.ball_history = []          # [(x, y, frame_idx), ...]
        self.last_shot_frame = -999     # para cooldown global
        print(f"[ShotClassifier] Detector por cambio de dirección iniciado (fps={fps})")

    # ------------------------------------------------------------------
    def detect_impact(self, frame_idx, ball_pos, players):
        """
        Detecta un golpe cuando la pelota cambia de dirección bruscamente.
        Devuelve dict con frame/pos/player_id o None.
        """
        if ball_pos is None:
            return None

        bx, by = float(ball_pos[0]), float(ball_pos[1])
        self.ball_history.append((bx, by, frame_idx))
        if len(self.ball_history) > self.HISTORY_SIZE:
            self.ball_history.pop(0)

        # Necesitamos suficientes puntos para tener antes + después del impacto
        needed = self.CHECK_OFFSET + self.VELOCITY_WINDOW + 1
        if len(self.ball_history) < needed:
            return None

        # Cooldown global
        if frame_idx - self.last_shot_frame < self.COOLDOWN_FRAMES:
            return None

        # Punto de impacto candidato: CHECK_OFFSET frames antes del actual
        impact_idx = len(self.ball_history) - 1 - self.CHECK_OFFSET
        if impact_idx < self.VELOCITY_WINDOW:
            return None

        impact_x, impact_y, impact_frame = self.ball_history[impact_idx]

        # Vectores de velocidad: promedio de VELOCITY_WINDOW frames antes y después
        before = self.ball_history[impact_idx - self.VELOCITY_WINDOW: impact_idx]
        after  = self.ball_history[impact_idx + 1: impact_idx + 1 + self.VELOCITY_WINDOW]

        if len(before) < 2 or len(after) < 2:
            return None

        # Velocidad media antes del impacto
        dt_b = max(1, before[-1][2] - before[0][2])
        vb = ((before[-1][0] - before[0][0]) / dt_b,
              (before[-1][1] - before[0][1]) / dt_b)

        # Velocidad media después del impacto
        dt_a = max(1, after[-1][2] - after[0][2])
        va = ((after[-1][0] - after[0][0]) / dt_a,
              (after[-1][1] - after[0][1]) / dt_a)

        speed_b = np.sqrt(vb[0]**2 + vb[1]**2)
        speed_a = np.sqrt(va[0]**2 + va[1]**2)

        # Ignorar si la pelota está casi parada (TrackNet confundido o pausa)
        if speed_b < 3.0 or speed_a < 3.0:
            return None

        # Ángulo entre los dos vectores de velocidad
        cos_a = np.clip((vb[0]*va[0] + vb[1]*va[1]) / (speed_b * speed_a), -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))

        # Debug cada 300 frames
        if frame_idx % 300 == 0:
            print(f"[ShotDebug] Frame {frame_idx}: ángulo={angle:.1f}° speed_b={speed_b:.1f} speed_a={speed_a:.1f}")

        if angle < self.MIN_ANGLE:
            return None

        # --- Golpe detectado — atribuir al jugador más cercano ---
        if not players:
            return None

        closest_pid  = 0
        closest_dist = float('inf')
        for px1, py1, px2, py2, pid in players:
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            d  = np.sqrt((impact_x - cx)**2 + (impact_y - cy)**2)
            if d < closest_dist:
                closest_dist = d
                closest_pid  = pid

        # Si no hay ningún jugador cerca → rebote en pared/red, no un golpe
        if closest_dist > self.MAX_PLAYER_DIST:
            if frame_idx % 100 == 0:
                print(f"[ShotDebug] Frame {frame_idx}: ángulo={angle:.1f}° RECHAZADO "
                      f"(jugador más cercano a {closest_dist:.0f}px > {self.MAX_PLAYER_DIST}px)")
            return None

        if closest_pid <= 0:
            return None

        print(f"[Shot] Frame {frame_idx}: GOLPE J{closest_pid} "
              f"(ángulo={angle:.1f}° dist={closest_dist:.0f}px impact_frame={impact_frame})")
        self.last_shot_frame = impact_frame
        return {
            "frame":     impact_frame,
            "pos":       (impact_x, impact_y),
            "player_id": closest_pid,
            "angle":     angle,
            "dist":      closest_dist,
        }

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
