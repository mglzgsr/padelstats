import numpy as np

class ShotClassifier:
    def __init__(self):
        self.ball_buffer = []  # List of (center_x, center_y, frame_idx)
        self.buffer_size = 15
        self.impact_threshold_angle = 90  # Degrees
        self.impact_threshold_speed_change = 1.5  # Ratio

    def detect_impact(self, frame_idx, ball_pos):
        """
        Detects a sudden change in ball trajectory or speed.
        ball_pos: (x, y) or None
        """
        if ball_pos is None:
            return None

        self.ball_buffer.append((ball_pos[0], ball_pos[1], frame_idx))
        if len(self.ball_buffer) > self.buffer_size:
            self.ball_buffer.pop(0)

        if len(self.ball_buffer) < 5:
            return None

        # Calculate vectors
        # Vector 1: before current point (last 2-3 points)
        # Vector 2: after current point (not possible in real-time without delay, 
        # but we can look at the sudden change to the current point)
        
        # Simpler: compare current velocity with previous average velocity
        if len(self.ball_buffer) >= 3:
            p1 = self.ball_buffer[-3]
            p2 = self.ball_buffer[-2]
            p3 = self.ball_buffer[-1]
            
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Check for sudden direction change
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 2 and mag2 > 2:
                cos_theta = np.dot(v1, v2) / (mag1 * mag2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                
                # If angle change is significant, it might be a hit or bounce
                if angle > self.impact_threshold_angle:
                    return {
                        "frame": frame_idx,
                        "pos": (p3[0], p3[1]),
                        "type": "potential_impact",
                        "angle": angle
                    }
        return None

    def classify_shot(self, impact, players):
        """
        Classifies the impact event based on proximity to players.
        impact: dict from detect_impact
        players: list of player boxes [(x1,y1,x2,y2,id), ...]
        """
        if not impact or not players:
            return None

        ix, iy = impact["pos"]
        
        # Find closest player
        closest_player = None
        min_dist = float('inf')
        
        for p in players:
            px1, py1, px2, py2, pid = p
            # Center of player
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dist = np.sqrt((ix - pcx)**2 + (iy - pcy)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = p

        # If impact is near a player, it's a shot. Otherwise it's a bounce on court/wall.
        # Strict threshold (80px) to ensure it's a racket hit, not a floor bounce near feet
        if closest_player and min_dist < 80:
            px1, py1, px2, py2, pid = closest_player
            p_height = py2 - py1
            
            # Simple classification logic
            shot_type = "Stroke" # Default
            
            # 1. Height check (Smash/Bandeja)
            # If hit is in top 20% of player height or above head
            if iy < py1 + p_height * 0.2:
                shot_type = "Smash/Bandeja"

            return {
                "frame": impact["frame"],
                "player_id": pid,
                "shot_type": shot_type,
                "pos": (ix, iy),
                "event": "Shot"
            }
        
        # If not near a player, classify as Bounce
        return {
            "frame": impact["frame"],
            "pos": (ix, iy),
            "event": "Bounce"
        }
