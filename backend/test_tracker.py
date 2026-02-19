"""
Script de prueba del tracker mejorado.
Procesa los primeros MAX_FRAMES frames del video y guarda el resultado.
"""
import cv2
import os
import numpy as np
from cv_engine.court_detector import CourtDetector
from cv_engine.tracker import Tracker

VIDEO_IN  = "uploaded_videos/match.mov"
VIDEO_OUT = "../debug_output/test_tracker_new.mp4"
MAX_FRAMES = 600  # ~10 segundos a 60fps

os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_IN)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

tracker = Tracker(model_path="yolov8n.pt", ball_model_path="tennis_ball_best.pt")
court_d = CourtDetector(config_path="court_config.json")

# Polígono inicial de respaldo (trapecio fijo) hasta que court_d detecte el azul
top_y_court = int(h * 0.45)
fallback_polygon = np.array([
    [int(w * 0.25), top_y_court],
    [int(w * 0.75), top_y_court],
    [w, h],
    [0, h]
], np.int32)
court_polygon = fallback_polygon  # Se reemplaza frame a frame con el detectado

colors = {1: (255, 80, 80), 2: (80, 80, 255), 3: (80, 255, 80), 4: (255, 255, 80), 0: (150, 150, 150)}

slot_changes   = {1: 0, 2: 0, 3: 0, 4: 0}
prev_yolo_ids  = {1: None, 2: None, 3: None, 4: None}
frame_count    = 0

print(f"Procesando primeros {MAX_FRAMES} frames de {VIDEO_IN}...")
print(f"NET_Y = {tracker.NET_Y}  ({int(h * tracker.NET_Y)}px en un frame de {h}px)")

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar polígono de pista por color azul (actualiza el polígono para el próximo frame)
    detected_poly = court_d.detect_court_polygon(frame)
    if detected_poly is not None:
        court_polygon = detected_poly

    results   = tracker.track_frame(frame, frame_count, court_polygon=court_polygon)
    p_results = results["person_results"]
    mapping   = p_results.slot_mapping if hasattr(p_results, "slot_mapping") else {}

    ann = frame.copy()

    # --- Línea de la red ---
    net_y = int(h * tracker.NET_Y)
    cv2.line(ann, (0, net_y), (w, net_y), (0, 165, 255), 2)
    cv2.putText(ann, f"RED (NET_Y={tracker.NET_Y})", (20, net_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Polígono de pista detectado (azul cian = detectado por color, gris = fallback)
    poly_color = (0, 220, 220) if detected_poly is not None else (80, 80, 80)
    cv2.polylines(ann, [court_polygon], True, poly_color, 2)

    if p_results.boxes and p_results.boxes.id is not None:
        boxes_raw = p_results.boxes.xyxy.cpu().numpy()
        ids_raw   = p_results.boxes.id.cpu().numpy()
        confs_raw = p_results.boxes.conf.cpu().numpy()

        # --- Filtro 1: solo jugadores dentro del polígono de cancha ---
        in_court = []
        for i, (xyxy, yid, conf) in enumerate(zip(boxes_raw, ids_raw, confs_raw)):
            x1, y1, x2, y2 = xyxy
            feet = ((x1 + x2) / 2, y2)
            if cv2.pointPolygonTest(court_polygon, feet, False) >= 0:
                in_court.append((xyxy, int(yid), conf))

        # --- Filtro 2: NMS entre detecciones solapadas (mismo jugador detectado dos veces) ---
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0

        # Ordenar por confianza descendente, suprimir solapados > 0.4
        in_court.sort(key=lambda x: x[2], reverse=True)
        kept = []
        for cand in in_court:
            suppressed = any(iou(cand[0], k[0]) > 0.4 for k in kept)
            if not suppressed:
                kept.append(cand)

        # --- Conteo de cambios de yolo_id por slot ---
        for xyxy, yid, conf in kept:
            sid = mapping.get(yid, 0)
            if sid > 0:
                if prev_yolo_ids[sid] is not None and prev_yolo_ids[sid] != yid:
                    slot_changes[sid] += 1
                prev_yolo_ids[sid] = yid

        # --- Dibujar jugadores filtrados ---
        for xyxy, yid, conf in kept:
            x1, y1, x2, y2 = map(int, xyxy)
            sid = mapping.get(yid, 0)
            color = colors[sid]
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
            cv2.putText(ann, f"J{sid} (yolo:{yid})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Marcar los pies
            feet_x = (x1 + x2) // 2
            cv2.circle(ann, (feet_x, y2), 5, color, -1)

        # --- Jugadores rechazados (fuera de pista) en gris tenue ---
        for xyxy, yid, conf in zip(boxes_raw, ids_raw, confs_raw):
            x1, y1, x2, y2 = xyxy
            feet = ((x1 + x2) / 2, y2)
            if cv2.pointPolygonTest(court_polygon, feet, False) < 0:
                cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (80, 80, 80), 1)
                cv2.putText(ann, f"FUERA yolo:{int(yid)}",
                            (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # --- Dibujar detecciones extra de zona lejana (magenta) ---
    for fx1, fy1, fx2, fy2, fconf in results.get("ball_far_detections", []):
        if fy1 < h * 0.22:
            continue
        cv2.rectangle(ann, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (255, 0, 255), 1)
        cv2.putText(ann, f"far {fconf:.2f}", (int(fx1), int(fy1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # --- Dibujar pelota (con filtros básicos equivalentes a processor.py) ---
    ball_results = results["ball_results"]
    if ball_results and ball_results.boxes:
        for box in ball_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            # Filtro ROI: ignorar 22% superior (focos a ~15%, techo)
            if y1 < h * 0.22:
                # Mostrar en rojo semi-transparente los que se rechazan por ROI
                cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 180), 1)
                cv2.putText(ann, "ROI", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)
                continue

            # Filtro tamaño: 2-55px
            bw, bh = x2 - x1, y2 - y1
            if max(bw, bh) < 2 or max(bw, bh) > 55:
                continue

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            # Recuadro cian = pasa filtros básicos
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(ann, f"ball {conf:.2f}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.circle(ann, (cx, cy), 4, (0, 200, 255), -1)

    # Líneas de cancha
    court_lines = court_d.detect(frame)
    ann = court_d.draw_lines(ann, court_lines)

    cv2.putText(ann, f"Frame {frame_count} | Cambios: {slot_changes}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(ann)
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"  {frame_count}/{MAX_FRAMES} frames  |  Cambios de ID: {slot_changes}")

cap.release()
out.release()

print(f"\n=== RESULTADO ===")
print(f"Frames procesados: {frame_count}")
print(f"Cambios de ID por slot: {slot_changes}")
total = sum(slot_changes.values())
print(f"Total cambios: {total}  ({'bueno' if total < 10 else 'revisar'})")
print(f"Video guardado en: {VIDEO_OUT}")
