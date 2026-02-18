from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
import uuid

class VideoRecord(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    status = Column(String, default="pending") # pending, processing, completed, error
    total_frames = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class ShotEvent(Base):
    __tablename__ = "shot_events"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"))
    frame = Column(Integer)
    timestamp = Column(Float) # Seconds
    player_id = Column(Integer)
    shot_type = Column(String) # Stroke, Smash/Bandeja
    pos_x = Column(Float)
    pos_y = Column(Float)
    
    video = relationship("VideoRecord", backref="shots")

class BounceEvent(Base):
    __tablename__ = "bounce_events"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"))
    frame = Column(Integer)
    pos_x = Column(Float)
    pos_y = Column(Float)

    video = relationship("VideoRecord", backref="bounces")
