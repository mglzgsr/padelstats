import numpy as np
from .ball_tracker import BallTracker

class ShotClassifier:
    """Detector de golpes basado en proximidad pelota-jugador."""

    def __init__(self, fps=30.0):
        self.fps = fps
        self.shot_cooldown = max(15, int(fps * 0.5))  # Reducido de 0.8s a 0.5s
        # Cooldown por jugador (no global)
        self.last_shot_frame = {1: -999, 2: -999, 3: -999, 4: -999, 0: -999}
        self.proximity_buffer = []
        self.PROXIMITY_BUFFER_SIZE = 30

        # Ball Tracker con Kalman Filter para interpolación
        self.ball_tracker = BallTracker(fps=int(fps))
        print(f"[ShotClassifier] Inicializado con interpolación Kalman (fps={fps})")

    def detect_impact(self, frame_idx, ball_pos, players):
        """Detecta golpe por approach→min→departure con interpolación Kalman."""
        if not players:
            return None

        # Actualizar ball tracker con detección (o None para interpolar)
        if ball_pos is not None:
            bx, by = ball_pos
            # Actualizar tracker con detección real (conf asumida 0.5)
            ball_tracked = self.ball_tracker.update(frame_idx, (bx, by, 0.5))
        else:
            # Intentar interpolar
            ball_tracked = self.ball_tracker.update(frame_idx, None)

        # Si no hay posición (ni real ni interpolada), salir
        if ball_tracked is None:
            return None

        bx, by, conf = ball_tracked

        # Jugador más cercano
        closest_player, min_dist = None, float('inf')
        for p in players:
            px1, py1, px2, py2, pid = p
            dist = np.sqrt((bx-(px1+px2)/2)**2 + (by-(py1+py2)/2)**2)
            if dist < min_dist:
                min_dist, closest_player = dist, p

        if not closest_player:
            return None

        px1, py1, px2, py2, pid = closest_player
        p_height = py2 - py1
        p_width  = px2 - px1
        # Con TrackNet tenemos la pelota en ~99% de frames, así que podemos
        # usar un threshold mucho más ajustado que con YOLO.
        # La pelota tiene que estar realmente cerca del jugador para contar como golpe.
        threshold = max(60, min(p_height * 0.40, 150))

        # Guardar posición trackeada (no raw) para poder calcular velocidad
        self.proximity_buffer.append((min_dist, pid, (bx, by), frame_idx))
        if len(self.proximity_buffer) > self.PROXIMITY_BUFFER_SIZE:
            self.proximity_buffer.pop(0)

        # Reducido a 5 para detectar golpes más rápidamente
        if len(self.proximity_buffer) < 5:
            return None

        recent = self.proximity_buffer[-12:]  # Reducido a 12 para ventana más pequeña
        min_idx = min(range(len(recent)), key=lambda i: recent[i][0])

        # Más permisivo: permite min_idx más cerca de los bordes
        if min_idx < 3 or min_idx > len(recent)-3 or recent[min_idx][0] > threshold:
            return None

        approach = [d for d,_,_,_ in recent[:min_idx][-4:]]
        departure = [d for d,_,_,_ in recent[min_idx+1:][:4]]

        # Reducido a 2 para ser más permisivo
        if len(approach) < 2 or len(departure) < 2:
            return None

        # Checks más permisivos: 0.90 en vez de 0.85, y 1.10 en vez de 1.15
        is_app = np.mean(approach[len(approach)//2:]) < np.mean(approach[:len(approach)//2]) * 0.90
        is_dep = np.mean(departure[len(departure)//2:]) > np.mean(departure[:len(departure)//2]) * 1.10

        if is_app and is_dep:
            _, min_pid, min_pos, min_frame = recent[min_idx]
            # Si player_id es 0 (no asignado), buscar el jugador con slot válido más cercano
            if min_pid == 0 and players:
                valid_players = [(px1,py1,px2,py2,pid) for px1,py1,px2,py2,pid in players if pid > 0]
                if valid_players:
                    # Asignar al jugador válido más cercano
                    bx, by = min_pos
                    min_pid = min(valid_players, key=lambda p: np.sqrt((bx-(p[0]+p[2])/2)**2 + (by-(p[1]+p[3])/2)**2))[4]
                else:
                    min_pid = 1  # Fallback a J1

            # Verificar cooldown POR JUGADOR
            last_frame_this_player = self.last_shot_frame.get(min_pid, -999)
            if min_frame - last_frame_this_player < self.shot_cooldown:
                return None  # Este jugador golpeó hace poco, ignorar

            print(f"[Shot] Frame {frame_idx}: GOLPE J{min_pid} (dist={recent[min_idx][0]:.1f}px)")
            shot_type = "Smash/Bandeja" if min_pos[1] < py1 + p_height*0.3 else "Stroke"
            self.last_shot_frame[min_pid] = min_frame
            return {"frame": min_frame, "pos": min_pos, "player_id": min_pid, "shot_type": shot_type, "type": "proximity"}
        return None

    def classify_shot(self, impact, players):
        return {"frame": impact["frame"], "player_id": impact["player_id"], "shot_type": impact["shot_type"], "pos":
impact["pos"], "event": "Shot"} if impact else None