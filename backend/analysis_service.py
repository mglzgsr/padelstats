from .cv_engine.processor import VideoProcessor
from .persistence_models import VideoRecord, ShotEvent, BounceEvent
from .database import SessionLocal, engine, Base
import os
from datetime import datetime

# Initialize tables
Base.metadata.create_all(bind=engine)

class AnalysisService:
    def __init__(self, db):
        self.db = db

    def analyze_video(self, video_id: str, file_path: str):
        """
        Background task to run CV processing and save results.
        """
        db = SessionLocal()
        try:
            # 1. Update status
            video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
            if not video:
                print(f"DEBUG: Video {video_id} not found in database.")
                return
            
            print(f"DEBUG: Starting analysis for video {video_id} ({video.filename})...")
            video.status = "processing"
            
            # Clear old events
            db.query(ShotEvent).filter(ShotEvent.video_id == video_id).delete()
            db.query(BounceEvent).filter(BounceEvent.video_id == video_id).delete()
            db.commit()

            # Ensure we use absolute path for input
            abs_file_path = os.path.abspath(file_path)
            if not os.path.exists(abs_file_path):
                print(f"DEBUG ERROR: Input file not found at {abs_file_path}")
                video.status = "error"
                db.commit()
                return

            def save_shot_realtime(shot_data):
                # Use a fresh local session for each shot to be safe in threads
                shot_db = SessionLocal()
                try:
                    db_shot = ShotEvent(
                        video_id=video_id,
                        frame=shot_data["frame"],
                        timestamp=shot_data["frame"] / 30.0,
                        player_id=shot_data["player_id"],
                        shot_type=shot_data["shot_type"],
                        pos_x=shot_data["pos"][0],
                        pos_y=shot_data["pos"][1]
                    )
                    shot_db.add(db_shot)
                    shot_db.commit()
                except Exception as e:
                    print(f"DEBUG ERROR: Failed to save shot realtime: {e}")
                finally:
                    shot_db.close()

            # 2. Setup output path (Absolute)
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(backend_dir)
            output_dir = os.path.join(project_root, "runs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{video_id}_annotated.mp4")

            print(f"DEBUG: Input path: {abs_file_path}")
            print(f"DEBUG: Output path: {output_path}")

            # 3. Initialize Processor
            court_config = os.path.join(backend_dir, "court_config.json")
            processor = VideoProcessor(
                abs_file_path, output_path,
                court_config_path=court_config if os.path.exists(court_config) else None
            )
            
            # 4. Fijar total_frames al inicio para mostrar progreso correcto
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(abs_file_path)
            video.total_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()
            db.commit()

            # 5. Run Analysis con callbacks de evento y progreso
            def save_progress(current: int, total: int):
                try:
                    video.processed_frames = current
                    db.commit()
                except Exception:
                    db.rollback()

            print(f"Iniciando análisis de {video.total_frames} frames…")
            processor.process(on_event=save_shot_realtime, on_progress=save_progress)
            print("Análisis finalizado.")

            # 6. Finalize status
            video.status = "completed"
            video.processed_frames = video.total_frames
            video.processed_at = datetime.utcnow()
            db.commit()
            print(f"DEBUG: Analysis for {video_id} successful.")

        except Exception as e:
            db.rollback()
            video.status = "error"
            db.commit()
            print(f"DEBUG ERROR: Analysis failed for video {video_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            db.close()
