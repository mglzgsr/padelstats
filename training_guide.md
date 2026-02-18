# Guía de Entrenamiento y Refinamiento del Modelo

Para alcanzar una fiabilidad profesional, puedes "enseñar" al modelo a distinguir mejor la pelota de los reflejos o las luces. Sigue estos pasos:

## 1. Extraer Imágenes de tus Vídeos
Usa el script que he creado para sacar los frames donde el modelo actual falla o tiene dudas:

```bash
# Extraer 200 frames aleatorios de tu partido
python3 scripts/extract_frames.py --video uploaded_videos/9b681e6b-17f2-439a-8234-372efab4adef.mov --num 200 --out dataset_padel

# O extraer "casos difíciles" (donde el modelo no está seguro)
python3 scripts/extract_frames.py --video match.mov --method hard_cases --model backend/tennis_ball_best.pt --num 100
```

## 2. Etiquetado en Roboflow (Recomendado)
Sube las imágenes de la carpeta `dataset_padel` a [Roboflow Project](https://roboflow.com):

1. **Clases**: Etiqueta únicamente la `ball` (pelota).
2. **Hard Negatives**: Si ves que el modelo confunde una luz del techo con la pelota, sube esa foto pero NO la etiquetes (déjala vacía). Esto enseña al modelo que "eso no es nada".
3. **Exportar**: Cuando tengas ~300 frames etiquetados, exporta el dataset en formato **YOLOv8**.

## 3. Retrenar el Modelo
Una vez tengas el dataset, puedes re-entrenar el modelo base:

```python
from ultralytics import YOLO

# Cargar el modelo que ya tenemos
model = YOLO('backend/tennis_ball_best.pt')

# Entrenar con tus nuevos datos
model.train(data='ruta/a/tu/data.yaml', epochs=50, imgsz=640)
```

## 4. Reemplazar y Probar
Copia el nuevo archivo `best.pt` generado a la carpeta `backend/` y reinicia el análisis. Notarás una mejora inmediata en la detección de la pelota y, por tanto, en la fiabilidad de los golpes detectados.
