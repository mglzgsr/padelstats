"""
Herramienta de calibración del polígono de pista.

Uso:
    python calibrate_court.py [ruta_video] [archivo_salida]

Defaults:
    video   = uploaded_videos/match.mov
    salida  = court_config.json

Instrucciones:
    - Haz clic en las ESQUINAS de la pista (suelo) en sentido horario
      empezando por la esquina superior izquierda.
    - Puedes usar cualquier número de puntos (mínimo 4).
    - ENTER  → guardar y salir
    - Z      → deshacer último punto
    - ESC    → salir sin guardar
    - Usa la rueda del ratón para hacer zoom (útil en esquinas difíciles)
"""
import cv2
import json
import sys
import os
import numpy as np

VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else "uploaded_videos/match.mov"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "court_config.json"
FRAME_IDX   = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # Frame a usar para calibrar

# ── Leer frame ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f"Error: no se puede leer el frame {FRAME_IDX} de {VIDEO_PATH}")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame {FRAME_IDX} cargado ({w}x{h})")
print("Instrucciones:")
print("  Clic izquierdo  → añadir punto")
print("  Z               → deshacer último punto")
print("  ENTER           → guardar y salir")
print("  ESC             → salir sin guardar")

# ── Estado ────────────────────────────────────────────────────────────────────
points = []

def draw(img):
    vis = img.copy()
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 7, (0, 255, 255), -1)
        cv2.circle(vis, pt, 7, (0, 0, 0), 2)
        cv2.putText(vis, str(i + 1), (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(points) >= 2:
        pts_arr = np.array(points, np.int32)
        cv2.polylines(vis, [pts_arr], len(points) > 2, (0, 200, 255), 2)
    n = len(points)
    msg = f"Puntos: {n}  |  ENTER=guardar  Z=deshacer  ESC=salir"
    cv2.putText(vis, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(vis, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    return vis

def on_mouse(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.imshow("Calibrar pista", draw(frame))

cv2.namedWindow("Calibrar pista", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibrar pista", min(w, 1400), min(h, 900))
cv2.setMouseCallback("Calibrar pista", on_mouse)
cv2.imshow("Calibrar pista", draw(frame))

# ── Bucle principal ───────────────────────────────────────────────────────────
saved = False
while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 13 or key == 10:  # ENTER
        if len(points) < 3:
            print("Necesitas al menos 3 puntos.")
        else:
            saved = True
            break
    elif key == ord('z') or key == ord('Z'):
        if points:
            points.pop()
            cv2.imshow("Calibrar pista", draw(frame))
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()

# ── Guardar ───────────────────────────────────────────────────────────────────
if saved:
    config = {
        "video": os.path.basename(VIDEO_PATH),
        "frame_width": w,
        "frame_height": h,
        "calibration_frame": FRAME_IDX,
        "court_polygon": points
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nGuardado en {OUTPUT_PATH}:")
    for i, pt in enumerate(points):
        print(f"  Punto {i+1}: {pt}")
    print(f"\nUsa este archivo pasándolo al VideoProcessor o al tracker.")
else:
    print("Cancelado, no se guardó nada.")
