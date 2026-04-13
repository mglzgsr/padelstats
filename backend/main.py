from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil
import os
import uuid
import tempfile
from datetime import datetime
from .database import get_db, engine, Base
from .persistence_models import VideoRecord, ShotEvent
from .analysis_service import AnalysisService
from .storage import storage, BACKEND, LOCAL_DIR

# Initialize DB
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Padel Stats API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir ficheros locales solo si el backend es local
if BACKEND == "local":
    app.mount("/files", StaticFiles(directory=str(LOCAL_DIR)), name="files")

@app.get("/")
def read_root():
    return {"message": "Padel Stats API", "storage": BACKEND}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    print(f"[UPLOAD] Iniciando upload: {file.filename}, tipo: {file.content_type}")

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename or ".mp4")[1]
    filename = f"{file_id}{file_extension}"
    print(f"[UPLOAD] file_id={file_id}, filename={filename}")

    try:
        # Guardar temporalmente y luego mover al backend de storage
        print(f"[UPLOAD] Guardando archivo temporal...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        print(f"[UPLOAD] Archivo temporal guardado en: {tmp_path}")

        print(f"[UPLOAD] Moviendo a storage...")
        stored_path = storage.save(tmp_path, filename)
        print(f"[UPLOAD] Archivo guardado en: {stored_path}")
        os.unlink(tmp_path)

        print(f"[UPLOAD] Guardando en base de datos...")
        db = next(get_db())
        db_video = VideoRecord(id=file_id, filename=file.filename, status="uploaded")
        db.add(db_video)
        db.commit()
        db.close()
        print(f"[UPLOAD] ✅ Upload completado: {file_id}")

    except Exception as e:
        print(f"[UPLOAD] ❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    return {
        "id": file_id,
        "filename": file.filename,
        "location": stored_path,
        "status": "uploaded",
        "upload_date": datetime.now().isoformat()
    }

@app.post("/analyze/{video_id}")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status == "processing":
        return {"message": "Analysis already in progress", "status": "processing"}

    # Limpiar resultados anteriores si es un reintento (status == "error" o "completed")
    if video.status in ("error", "completed"):
        db.query(ShotEvent).filter(ShotEvent.video_id == video_id).delete()
        video.processed_frames = 0
        video.total_frames = 0
        db.commit()

    video.status = "processing"
    db.commit()

    # Construir path al vídeo según el backend de storage
    file_extension = os.path.splitext(video.filename)[1]
    if BACKEND == "local":
        file_path = os.path.join(str(LOCAL_DIR), f"{video.id}{file_extension}")
    else:
        # Para S3: la key es uploads/{filename}
        file_path = f"uploads/{video.id}{file_extension}"

    service = AnalysisService(db)
    background_tasks.add_task(service.analyze_video, video.id, file_path)

    return {"message": "Analysis started", "video_id": video.id, "status": "processing"}

@app.get("/results/{video_id}")
def get_results(video_id: str, db: Session = Depends(get_db)):
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    shots = db.query(ShotEvent).filter(ShotEvent.video_id == video_id).all()
    
    # Aggregate some stats
    player_stats = {}
    for shot in shots:
        pid = shot.player_id
        if pid not in player_stats:
            player_stats[pid] = {"total_shots": 0, "smash_count": 0, "stroke_count": 0}
        
        player_stats[pid]["total_shots"] += 1
        if shot.shot_type == "Smash/Bandeja":
            player_stats[pid]["smash_count"] += 1
        else:
            player_stats[pid]["stroke_count"] += 1

    return {
        "video_id": video_id,
        "filename": video.filename,
        "status": video.status,
        "stage": video.stage or "",
        "total_frames": video.total_frames or 0,
        "processed_frames": video.processed_frames or 0,
        "shots_count": len(shots),
        "player_stats": player_stats,
        "events": shots
    }

@app.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    videos = db.query(VideoRecord).all()
    return videos

@app.post("/cancel/{video_id}")
def cancel_analysis(video_id: str, db: Session = Depends(get_db)):
    """
    Cancela un análisis en proceso marcándolo como 'error'.
    Nota: el proceso de análisis seguirá corriendo en background,
    pero el usuario puede reiniciarlo después.
    """
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status != "processing":
        raise HTTPException(status_code=400, detail="Video is not being processed")

    video.status = "error"
    db.commit()

    return {"message": "Analysis cancelled", "video_id": video_id}

