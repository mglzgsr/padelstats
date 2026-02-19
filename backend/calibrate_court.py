"""
Herramienta de calibración de pista de pádel (cámara fija).

Fases:
  1. Polígono de pista  – clic libre en las esquinas del suelo
  2. Líneas pintadas    – 2 clics por línea (inicio y fin)
  3. Paredes / red      – 2 clics por elemento

Controles:
  Clic izquierdo  → añadir punto
  Z               → deshacer último punto de la fase actual
  ESPACIO/ENTER   → confirmar fase actual y pasar a la siguiente
  ESC             → salir sin guardar

Uso:
  python calibrate_court.py [video] [salida]
  Defaults: uploaded_videos/match.mov  court_config.json
"""
import cv2
import json
import sys
import os
import numpy as np

VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else "uploaded_videos/match.mov"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "court_config.json"
FRAME_IDX   = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# ── Leer frame ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f"Error: no se puede leer el frame {FRAME_IDX} de {VIDEO_PATH}")
    sys.exit(1)

H, W = frame.shape[:2]
print(f"Frame {FRAME_IDX} cargado ({W}×{H})")

# ── Definición de fases ───────────────────────────────────────────────────────
# type "polygon": clic libre, ESPACIO/ENTER para confirmar (mín. min_pts puntos)
# type "line":    exactamente 2 clics, avanza automáticamente
PHASES = [
    {
        "key":         "court_polygon",
        "type":        "polygon",
        "min_pts":     4,
        "color":       (0, 200, 255),
        "title":       "FASE 1/3 — Polígono de pista",
        "instruction": "Clic en las ESQUINAS del suelo (min. 4). ESPACIO/ENTER para continuar.",
    },
    {
        "key":         "line_service_far",
        "type":        "line",
        "color":       (0, 255, 80),
        "title":       "FASE 2/3 — Línea de saque FONDO (lado red)",
        "instruction": "2 clics: extremo izquierdo → derecho de la línea de saque del fondo.",
    },
    {
        "key":         "line_service_near",
        "type":        "line",
        "color":       (0, 255, 80),
        "title":       "FASE 2/3 — Línea de saque CÁMARA (lado cercano)",
        "instruction": "2 clics: extremo izquierdo → derecho de la línea de saque cercana.",
    },
    {
        "key":         "line_center",
        "type":        "line",
        "color":       (0, 255, 200),
        "title":       "FASE 2/3 — Línea central de saque (vertical)",
        "instruction": "2 clics: extremo superior → inferior de la línea central.",
    },
    {
        "key":         "wall_left",
        "type":        "line",
        "color":       (255, 120, 0),
        "title":       "FASE 3/3 — Pared lateral IZQUIERDA",
        "instruction": "2 clics: unión suelo-pared izq. (punto cercano → punto lejano).",
    },
    {
        "key":         "wall_right",
        "type":        "line",
        "color":       (255, 120, 0),
        "title":       "FASE 3/3 — Pared lateral DERECHA",
        "instruction": "2 clics: unión suelo-pared der. (punto cercano → punto lejano).",
    },
    {
        "key":         "net",
        "type":        "line",
        "color":       (80, 80, 255),
        "title":       "FASE 3/3 — Red",
        "instruction": "2 clics: extremo izquierdo → derecho de la red (al nivel del suelo).",
    },
]

# ── Estado ────────────────────────────────────────────────────────────────────
phase_idx   = 0          # fase actual
collected   = {}         # key → lista de puntos completados
current_pts = []         # puntos de la fase actual (en curso)
done        = False
cancelled   = False


def draw_all():
    vis = frame.copy()

    # Dibujar elementos ya completados
    for ph in PHASES[:phase_idx]:
        pts = collected.get(ph["key"], [])
        color = ph["color"]
        if ph["type"] == "polygon" and len(pts) >= 2:
            arr = np.array(pts, np.int32)
            cv2.polylines(vis, [arr], True, color, 2)
            for p in pts:
                cv2.circle(vis, p, 5, color, -1)
        elif ph["type"] == "line" and len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], color, 2)
            for p in pts:
                cv2.circle(vis, p, 5, color, -1)

    # Dibujar fase actual (en progreso)
    ph = PHASES[phase_idx]
    color_cur = ph["color"]
    if ph["type"] == "polygon":
        for p in current_pts:
            cv2.circle(vis, p, 7, color_cur, -1)
            cv2.circle(vis, p, 7, (0, 0, 0), 2)
        if len(current_pts) >= 2:
            arr = np.array(current_pts, np.int32)
            cv2.polylines(vis, [arr], False, color_cur, 2)
    elif ph["type"] == "line":
        for p in current_pts:
            cv2.circle(vis, p, 7, color_cur, -1)
            cv2.circle(vis, p, 7, (0, 0, 0), 2)
        if len(current_pts) == 2:
            cv2.line(vis, current_pts[0], current_pts[1], color_cur, 2)

    # HUD: título e instrucción
    _text_bg(vis, ph["title"],       (15, 40),  0.80, (200, 255, 200))
    _text_bg(vis, ph["instruction"], (15, 75),  0.60, (255, 255, 255))
    _text_bg(vis, "Z=deshacer  ESPACIO/ENTER=confirmar  ESC=salir",
             (15, H - 20), 0.55, (200, 200, 200))
    return vis


def _text_bg(img, text, org, scale, color):
    """Texto con fondo negro para legibilidad."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def advance_phase():
    """Guarda los puntos actuales y pasa a la siguiente fase (o termina)."""
    global phase_idx, current_pts, done
    collected[PHASES[phase_idx]["key"]] = list(current_pts)
    current_pts = []
    phase_idx += 1
    if phase_idx >= len(PHASES):
        done = True


def on_mouse(event, x, y, _flags, _param):
    global current_pts
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ph = PHASES[phase_idx]
    current_pts.append((x, y))
    # Fase "line": avanzar automáticamente al tener 2 puntos
    if ph["type"] == "line" and len(current_pts) == 2:
        advance_phase()
        if done:
            return
    cv2.imshow("Calibrar pista", draw_all())


# ── Ventana principal ─────────────────────────────────────────────────────────
WIN = "Calibrar pista"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, min(W, 1400), min(H, 900))
cv2.setMouseCallback(WIN, on_mouse)
cv2.imshow(WIN, draw_all())

while not done and not cancelled:
    key = cv2.waitKey(20) & 0xFF
    ph  = PHASES[phase_idx]

    if key == 27:          # ESC
        cancelled = True

    elif key in (13, 10, 32):   # ENTER / SPACE
        if ph["type"] == "polygon":
            if len(current_pts) < ph["min_pts"]:
                print(f"Necesitas al menos {ph['min_pts']} puntos.")
            else:
                advance_phase()
        # Para "line", el avance es automático al 2.º clic

    elif key in (ord('z'), ord('Z')):
        if current_pts:
            current_pts.pop()
            cv2.imshow(WIN, draw_all())

cv2.destroyAllWindows()

# ── Guardar ───────────────────────────────────────────────────────────────────
if cancelled:
    print("Cancelado, no se guardó nada.")
    sys.exit(0)

# Leer config existente para no perder datos previos
config = {}
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            pass

config.update({
    "video":             os.path.basename(VIDEO_PATH),
    "frame_width":       W,
    "frame_height":      H,
    "calibration_frame": FRAME_IDX,
    "court_polygon":     collected.get("court_polygon", config.get("court_polygon", [])),
    "court_lines": {
        "service_line_far":  collected.get("line_service_far",  config.get("court_lines", {}).get("service_line_far")),
        "service_line_near": collected.get("line_service_near", config.get("court_lines", {}).get("service_line_near")),
        "center_line":       collected.get("line_center",       config.get("court_lines", {}).get("center_line")),
    },
    "walls": {
        "left":  collected.get("wall_left",  config.get("walls", {}).get("left")),
        "right": collected.get("wall_right", config.get("walls", {}).get("right")),
        "net":   collected.get("net",        config.get("walls", {}).get("net")),
    },
})

with open(OUTPUT_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nGuardado en {OUTPUT_PATH}")
for key, val in collected.items():
    print(f"  {key}: {val}")
print("\nUsa court_config.json pasándolo al VideoProcessor.")
