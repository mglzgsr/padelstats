from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
import uuid

# Enums
ShotType = Literal["forehand", "backhand", "volley", "smash", "lob", "serve"]
Outcome = Literal["winner", "error", "in_play"]
Side = Literal["left", "right"]

class Player(BaseModel):
    id: str
    name: str
    ranking: Optional[int] = None

class Stat(BaseModel):
    # Aggregated stats
    forehand_winners: int = 0
    forehand_errors: int = 0
    backhand_winners: int = 0
    backhand_errors: int = 0
    volley_winners: int = 0
    volley_errors: int = 0
    smash_winners: int = 0
    smash_errors: int = 0
    total_points_won: int = 0
    total_points_lost: int = 0
    unforced_errors: int = 0
    
class PlayerMatchStats(BaseModel):
    player_id: str
    match_id: str
    stats: Stat

class Event(BaseModel):
    id: str = str(uuid.uuid4())
    match_id: str
    timestamp: float  # Seconds from start of video
    player_id: Optional[str] = None
    shot_type: Optional[ShotType] = None
    outcome: Optional[Outcome] = None
    coordinates_x: Optional[float] = None
    coordinates_y: Optional[float] = None

class Video(BaseModel):
    id: str
    filename: str
    duration: float
    upload_date: datetime
    processed: bool = False

class Match(BaseModel):
    id: str = str(uuid.uuid4())
    date: datetime
    video_id: Optional[str] = None
    players: List[Player]
    events: List[Event] = []
