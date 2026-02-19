from __future__ import annotations

import json
import os
import cv2
import numpy as np


class CourtDetector:
    def __init__(self, config_path: str | None = None):
        """
        config_path: ruta al JSON generado por calibrate_court.py.
            Si existe, el polígono de pista queda fijo (cámara fija).
            Si no, se detecta automáticamente por color azul cada frame.
        """
        self.court_lines = None
        # Polígono de pista estabilizado (se actualiza con media exponencial)
        self._stable_polygon = None
        self._poly_alpha = 0.05  # Actualización lenta para la detección automática
        self._polygon_fixed = False  # True cuando viene de calibración manual

        # Intentar cargar calibración manual
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Carga el polígono de pista desde un JSON de calibración.

        Las esquinas cercanas a la cámara suelen estar fuera del encuadre,
        por lo que se añaden automáticamente las esquinas inferiores del frame
        antes de calcular el casco convexo final.
        """
        with open(config_path) as f:
            config = json.load(f)
        pts = config.get("court_polygon", [])
        if len(pts) < 3:
            return

        w = config.get("frame_width", 1920)
        h = config.get("frame_height", 1080)

        # Añadir esquinas inferiores del frame para cubrir la zona cercana a cámara
        all_pts = np.array(pts + [[0, h - 1], [w - 1, h - 1]], dtype=np.float32)
        hull = cv2.convexHull(all_pts)
        self._stable_polygon = hull.reshape(-1, 2).astype(np.int32)
        self._polygon_fixed = True
        print(f"[CourtDetector] Polígono cargado desde {config_path} "
              f"({len(pts)} puntos + esquinas inferiores) — detección automática desactivada.")

    # ------------------------------------------------------------------
    # Detección del polígono de pista (suelo azul)
    # ------------------------------------------------------------------

    def detect_court_polygon(self, frame) -> np.ndarray | None:
        """
        Devuelve el polígono de la pista. Si hay calibración manual (JSON),
        lo retorna directamente sin procesar el frame. Si no, lo detecta
        automáticamente por color azul HSV.
        """
        # Si la calibración es manual, no hay nada que detectar
        if self._polygon_fixed:
            return self._stable_polygon

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango HSV para azul de pista de pádel
        # Cubre azul oscuro, azul medio y azul brillante bajo distintas luces
        lower_blue = np.array([95, 60, 40])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Ignorar la franja superior del frame (techo, tribuna, fondo)
        blue_mask[:int(h * 0.30), :] = 0

        # Morfología: cerrar huecos (líneas blancas dentro de la pista)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)

        # Eliminar ruido pequeño
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)

        # Encontrar el contorno más grande (la pista)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._stable_polygon  # Retorna el último polígono estable

        court_contour = max(contours, key=cv2.contourArea)

        # Filtro mínimo: la pista debe ocupar al menos el 10% del frame
        if cv2.contourArea(court_contour) < w * h * 0.10:
            return self._stable_polygon

        # Simplificar el contorno a un polígono convexo de pocos vértices
        hull = cv2.convexHull(court_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, epsilon, True)
        polygon = polygon.reshape(-1, 2).astype(np.float32)

        # Los jugadores cercanos a la cámara están ENTRE la cámara y la pista,
        # por lo que sus pies aparecen debajo del área azul visible.
        # Forzamos que el polígono siempre incluya las esquinas inferiores del frame
        # para que nunca se excluya a los jugadores del lado cercano.
        bottom_corners = np.array([[0, h - 1], [w - 1, h - 1]], dtype=np.float32)
        polygon = np.vstack([polygon, bottom_corners])
        hull_extended = cv2.convexHull(polygon.astype(np.float32))
        polygon = hull_extended.reshape(-1, 2).astype(np.float32)

        # Estabilizar el polígono con media exponencial para evitar saltos
        if self._stable_polygon is None or len(self._stable_polygon) != len(polygon):
            self._stable_polygon = polygon.astype(np.int32)
        else:
            self._stable_polygon = (
                (1 - self._poly_alpha) * self._stable_polygon + self._poly_alpha * polygon
            ).astype(np.int32)

        return self._stable_polygon

    def get_stable_polygon(self) -> np.ndarray | None:
        """Devuelve el último polígono estabilizado (para usar como máscara de cancha)."""
        return self._stable_polygon

    # ------------------------------------------------------------------
    # Detección de líneas internas
    # ------------------------------------------------------------------

    def detect(self, frame, debug=False):
        """
        Detecta las líneas internas de la pista: líneas de servicio (horizontales),
        línea central de saque (vertical) y bandas laterales (diagonales).
        Los segmentos Hough se fusionan por regresión lineal para obtener
        líneas limpias y extendidas en lugar de decenas de trozos cortos.
        """
        h, w = frame.shape[:2]

        # ROI: polígono de pista o trapecio de respaldo
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        if self._stable_polygon is not None:
            cv2.fillPoly(roi_mask, [self._stable_polygon], 255)
        else:
            top_y = int(h * 0.50)
            pts = np.array([
                [int(w * 0.15), top_y], [int(w * 0.85), top_y], [w, h], [0, h]
            ], np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)

        # Máscara HSV para líneas blancas
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        combined_mask = cv2.bitwise_and(white_mask, white_mask, mask=roi_mask)

        # Morfología: dilatar para conectar trozos interrumpidos por la perspectiva
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        # Hough — umbral más bajo y gap mayor para capturar líneas parciales
        raw_lines = cv2.HoughLinesP(
            combined_mask, 1, np.pi / 180,
            threshold=60, minLineLength=60, maxLineGap=80
        )

        if raw_lines is None:
            self.court_lines = None
            return None

        # Clasificar segmentos por tipo de línea
        h_segs = []   # horizontales (líneas de servicio)
        v_segs = []   # verticales (centro de saque)
        d_segs = []   # diagonales (bandas laterales)

        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            if angle < 25 or angle > 155:
                # Horizontal: líneas de servicio en el área de juego
                if h * 0.45 < mid_y < h * 0.95:
                    h_segs.append((x1, y1, x2, y2))
            elif 65 < angle < 115:
                # Vertical: línea central de saque, evitar bordes laterales
                if w * 0.30 < mid_x < w * 0.70:
                    v_segs.append((x1, y1, x2, y2))
            elif 25 <= angle <= 65 or 115 <= angle <= 155:
                # Diagonal: bandas laterales visibles por perspectiva
                # Solo en los bordes laterales de la pista
                if mid_x < w * 0.25 or mid_x > w * 0.75:
                    if h * 0.40 < mid_y < h * 0.90:
                        d_segs.append((x1, y1, x2, y2))

        # Fusionar segmentos del mismo tipo en líneas limpias
        final_lines = []
        # Horizontales: agrupar por Y (± 40px) → máx. 3 líneas
        for merged in self._cluster_and_merge(h_segs, axis="y", gap=40, max_lines=3):
            final_lines.append([list(merged)])
        # Vertical central: máx. 1 línea
        for merged in self._cluster_and_merge(v_segs, axis="x", gap=60, max_lines=1):
            final_lines.append([list(merged)])
        # Bandas laterales: máx. 2 líneas (izquierda + derecha)
        for merged in self._cluster_and_merge(d_segs, axis="x", gap=80, max_lines=2):
            final_lines.append([list(merged)])

        lines = np.array(final_lines, dtype=np.int32) if final_lines else None
        self.court_lines = lines
        return lines

    def _cluster_and_merge(self, segs: list, axis: str, gap: int, max_lines: int) -> list:
        """
        Agrupa segmentos por proximidad en el eje indicado y fusiona cada grupo
        en una sola línea extendida mediante regresión lineal.
        Devuelve lista de (x1, y1, x2, y2) con los mejores max_lines grupos.
        """
        if not segs:
            return []

        # Ordenar por el punto medio del eje de agrupación
        key_fn = (lambda s: (s[1] + s[3]) / 2) if axis == "y" \
            else (lambda s: (s[0] + s[2]) / 2)
        segs = sorted(segs, key=key_fn)

        # Agrupar segmentos consecutivos dentro del gap
        clusters = []
        current = [segs[0]]
        for seg in segs[1:]:
            if abs(key_fn(seg) - key_fn(current[-1])) <= gap:
                current.append(seg)
            else:
                clusters.append(current)
                current = [seg]
        clusters.append(current)

        # Ordenar clusters por número de segmentos (más largo = más fiable)
        clusters.sort(key=lambda c: len(c), reverse=True)

        merged = []
        for cluster in clusters[:max_lines]:
            pts = []
            for x1, y1, x2, y2 in cluster:
                pts.extend([(x1, y1), (x2, y2)])
            pts = np.array(pts, dtype=np.float32)

            xs, ys = pts[:, 0], pts[:, 1]
            # Regresión lineal: si más dispersión en X → y=f(x), si no → x=f(y)
            if np.std(xs) >= np.std(ys):
                if np.std(xs) < 1:
                    continue
                m, b = np.polyfit(xs, ys, 1)
                x1e, x2e = int(xs.min()), int(xs.max())
                y1e, y2e = int(m * x1e + b), int(m * x2e + b)
            else:
                if np.std(ys) < 1:
                    continue
                m, b = np.polyfit(ys, xs, 1)
                y1e, y2e = int(ys.min()), int(ys.max())
                x1e, x2e = int(m * y1e + b), int(m * y2e + b)

            merged.append((x1e, y1e, x2e, y2e))

        return merged

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def draw_lines(self, frame, lines=None):
        if lines is None:
            lines = self.court_lines
        if lines is not None:
            h, w = frame.shape[:2]
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 25 or angle > 155:
                    color = (0, 255, 0)    # verde = horizontal (servicio)
                elif 65 < angle < 115:
                    color = (0, 200, 255)  # amarillo = vertical (centro)
                else:
                    color = (255, 180, 0)  # azul = banda lateral
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    def draw_court_polygon(self, frame, color=(0, 200, 255), thickness=2):
        """Dibuja el contorno de la pista detectado por color azul."""
        if self._stable_polygon is not None:
            cv2.polylines(frame, [self._stable_polygon], True, color, thickness)
        return frame
