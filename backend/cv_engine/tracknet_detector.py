"""
TrackNetDetector — integración de TrackNetV3 fine-tuned como detector de pelota.

Workflow:
1. Escala el vídeo a 960x540 (resolución de entrenamiento)
2. Lanza predict.py de TrackNetV3 como subprocess
3. Lee el CSV de salida
4. Devuelve posiciones escaladas a la resolución original del vídeo

Configuración via variables de entorno:
  TRACKNET_DIR     — ruta al repo TrackNetV3 clonado (con predict.py)
  TRACKNET_MODEL   — ruta al .pt fine-tuned de TrackNet
  INPAINTNET_MODEL — ruta al .pt de InpaintNet (opcional, mejora oclusiones)

Si las variables no están definidas o el modelo no existe, available=False
y el sistema cae al detector YOLO como hasta ahora.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple


class TrackNetDetector:
    MODEL_W, MODEL_H = 960, 540   # resolución de entrenamiento

    def __init__(self):
        self.tracknet_dir   = os.environ.get('TRACKNET_DIR', '')
        self.model_path     = os.environ.get('TRACKNET_MODEL', '')
        self.inpaintnet     = os.environ.get('INPAINTNET_MODEL', '')

        self.available = bool(
            self.tracknet_dir
            and os.path.isfile(os.path.join(self.tracknet_dir, 'predict.py'))
            and os.path.isfile(self.model_path)
        )

        if self.available:
            print(f'[TrackNet] Detector disponible — {self.model_path}')
        else:
            print('[TrackNet] No configurado — usando YOLO para pelota')

    # ------------------------------------------------------------------
    def detect_video(
        self,
        video_path: str,
        video_w: int,
        video_h: int,
        on_progress=None,
    ) -> Dict[int, Tuple[float, float, bool]]:
        """
        Lanza TrackNetV3 sobre video_path y devuelve posiciones de pelota.

        Returns:
            {frame_idx: (x, y, visible)}
            — coordenadas escaladas a la resolución original (video_w × video_h)
            — visible=False significa pelota no detectada en ese frame
        """
        if not self.available:
            return {}

        if on_progress:
            on_progress('tracknet_start', 0)

        with tempfile.TemporaryDirectory() as tmp:
            scaled   = os.path.join(tmp, 'scaled.mp4')
            pred_dir = os.path.join(tmp, 'pred')
            os.makedirs(pred_dir)

            # 1. Escalar al tamaño de entrenamiento
            print(f'[TrackNet] Escalando vídeo a {self.MODEL_W}x{self.MODEL_H}...')
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'scale={self.MODEL_W}:{self.MODEL_H}',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-an',
                scaled
            ], check=True, capture_output=True)

            if on_progress:
                on_progress('tracknet_scaled', 10)

            # 2. Parchear predict.py para CPU/MPS y copiarlo a /tmp
            #    (no modificar in-place: uvicorn --reload detectaría el cambio y recargaría)
            import re as _re
            predict_py = os.path.join(self.tracknet_dir, 'predict.py')
            with open(predict_py) as f:
                src = f.read()

            # Forzar CPU: MPS puede disparar jetsam (presión de memoria) en macOS
            # antes de que el modelo siquiera cargue, causando SIGTERM (-15).
            _device = 'cpu'

            patched = src
            # map_location en torch.load
            patched = patched.replace(
                'torch.load(args.tracknet_file)',
                f"torch.load(args.tracknet_file, map_location='{_device}')"
            ).replace(
                'torch.load(args.inpaintnet_file)',
                f"torch.load(args.inpaintnet_file, map_location='{_device}')"
            )
            # Reemplazar .cuda() por .to(device)
            patched = _re.sub(r'\.cuda\(\)', f".to('{_device}')", patched)
            # Reemplazar device='cuda' o device="cuda"
            patched = _re.sub(r"device=['\"]cuda['\"]", f"device='{_device}'", patched)

            # Inyectar sys.path para que los imports relativos de TrackNetV3 funcionen
            # (Python añade el dir del script a sys.path, no cwd, así que hay que añadirlo explícitamente)
            path_injection = (
                f"import sys as _sys\n"
                f"if {repr(self.tracknet_dir)} not in _sys.path:\n"
                f"    _sys.path.insert(0, {repr(self.tracknet_dir)})\n"
            )
            patched = path_injection + patched

            # Guardar en /tmp (fuera del directorio vigilado por --reload)
            patched_predict = os.path.join(tmp, 'predict_patched.py')
            with open(patched_predict, 'w') as f:
                f.write(patched)
            print(f'[TrackNet] predict.py parcheado → device={_device}')

            # 3. Lanzar predict.py parcheado desde /tmp, con cwd=tracknet_dir
            #    para que los imports relativos de TrackNetV3 funcionen
            print('[TrackNet] Corriendo inferencia...')
            cmd = [
                sys.executable, patched_predict,
                '--video_file',    scaled,
                '--tracknet_file', self.model_path,
                '--save_dir',      pred_dir,
                '--large_video',
            ]
            if self.inpaintnet and os.path.isfile(self.inpaintnet):
                cmd += ['--inpaintnet_file', self.inpaintnet]

            ret = subprocess.run(
                cmd,
                cwd=self.tracknet_dir,
                capture_output=True,       # capturar para debug
                start_new_session=True,    # aísla del grupo de procesos de uvicorn
            )
            # Mostrar siempre stdout/stderr en el log para poder debugar
            if ret.stdout:
                print('[TrackNet stdout]', ret.stdout.decode(errors='replace'))
            if ret.stderr:
                print('[TrackNet stderr]', ret.stderr.decode(errors='replace'))
            if ret.returncode != 0:
                print(f'[TrackNet] predict.py falló (returncode={ret.returncode}) — usando YOLO')
                return {}

            if on_progress:
                on_progress('tracknet_done', 90)

            # 3. Leer CSV de salida
            csvs = list(Path(pred_dir).glob('*.csv'))
            if not csvs:
                print('[TrackNet] No se generó CSV — usando YOLO')
                return {}

            positions: Dict[int, Tuple[float, float, bool]] = {}
            scale_x = video_w / self.MODEL_W
            scale_y = video_h / self.MODEL_H

            with open(csvs[0], newline='') as f:
                for row in csv.DictReader(f):
                    frame = int(row['Frame'])
                    vis   = int(row['Visibility']) == 1
                    x     = float(row['X']) * scale_x if vis else 0.0
                    y     = float(row['Y']) * scale_y if vis else 0.0
                    positions[frame] = (x, y, vis)

            detected = sum(1 for _, _, v in positions.values() if v)
            total    = len(positions)
            rate     = detected / total * 100 if total else 0
            print(f'[TrackNet] {detected}/{total} frames con pelota ({rate:.1f}%)')

            if on_progress:
                on_progress('tracknet_loaded', 100)

            return positions
