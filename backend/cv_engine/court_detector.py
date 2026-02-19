from __future__ import annotations

import cv2
import numpy as np


class CourtDetector:
    def __init__(self):
        self.court_lines = None
        # Polígono de pista estabilizado (se actualiza con media exponencial)
        self._stable_polygon = None
        self._poly_alpha = 0.05  # Actualización lenta: el polígono no salta frame a frame

    # ------------------------------------------------------------------
    # Detección del polígono de pista (suelo azul)
    # ------------------------------------------------------------------

    def detect_court_polygon(self, frame) -> np.ndarray | None:
        """
        Detecta el contorno de la pista usando el color azul del suelo.
        Devuelve un polígono np.int32 (N,2) en coordenadas de imagen,
        o None si no se puede detectar.

        La pista de pádel tiene suelo azul, lo que facilita una segmentación
        HSV robusta frente al fondo (paredes de cristal, suelo exterior verde).
        """
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
        Detecta las 3 líneas internas principales: 2 horizontales + 1 vertical central.
        Usa el polígono azul como ROI para limitar la búsqueda al suelo de la pista.
        """
        h, w = frame.shape[:2]

        # ROI: usar el polígono de pista si está disponible, si no el trapecio fijo
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        if self._stable_polygon is not None:
            cv2.fillPoly(roi_mask, [self._stable_polygon], 255)
        else:
            top_y = int(h * 0.55)
            pts = np.array([
                [int(w * 0.2), top_y], [int(w * 0.8), top_y], [w, h], [0, h]
            ], np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)

        # Máscara HSV para líneas blancas
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 185])
        upper_white = np.array([180, 45, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        combined_mask = cv2.bitwise_and(white_mask, white_mask, mask=roi_mask)

        # Morfología
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        # Hough Lines
        lines = cv2.HoughLinesP(
            combined_mask, 1, np.pi / 180,
            threshold=80, minLineLength=100, maxLineGap=50
        )

        if lines is not None:
            final_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Líneas horizontales (líneas de servicio)
                # Ampliar tolerancia angular: perspectiva hace que no sean exactamente 0°
                if angle < 20 or angle > 160:
                    if h * 0.55 < (y1 + y2) / 2 < h * 0.95:
                        final_lines.append(line)

                # Línea vertical central (línea de saque)
                elif 70 < angle < 110:
                    if w * 0.35 < (x1 + x2) / 2 < w * 0.65:
                        final_lines.append(line)

            lines = np.array(final_lines) if final_lines else None

        self.court_lines = lines
        return lines

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def draw_lines(self, frame, lines=None):
        if lines is None:
            lines = self.court_lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frame

    def draw_court_polygon(self, frame, color=(0, 200, 255), thickness=2):
        """Dibuja el contorno de la pista detectado por color azul."""
        if self._stable_polygon is not None:
            cv2.polylines(frame, [self._stable_polygon], True, color, thickness)
        return frame
