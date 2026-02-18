from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from datetime import datetime
from .database import get_db, engine, Base
from .persistence_models import VideoRecord, ShotEvent
from .analysis_service import AnalysisService

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

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to Padel Stats API"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{file_id}{file_extension}"
    file_location = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Save to DB
        db = next(get_db())
        db_video = VideoRecord(id=file_id, filename=file.filename)
        db.add(db_video)
        db.commit()
        db.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
        
    return {
        "id": file_id,
        "filename": file.filename,
        "location": file_location,
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
    
    # Run in background
    file_path = os.path.join(UPLOAD_DIR, f"{video.id}{os.path.splitext(video.filename)[1]}")
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
        "status": video.status,
        "total_frames": video.total_frames,
        "shots_count": len(shots),
        "player_stats": player_stats,
        "events": shots
    }

@app.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    videos = db.query(VideoRecord).all()
    return videos

