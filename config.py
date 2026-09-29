# config.py

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # File paths
    STUDENTS_FILE: str = "data/students.csv"
    ENCODINGS_FILE: str = "data/encodings.pkl"
    ATTENDANCE_FILE: str = "data/attendance.csv"
    
    # Campus location (latitude, longitude)
    CAMPUS_LOCATION: Tuple[float, float] = (23.2599, 77.4126)  # Example: New Delhi
    ALLOWED_RADIUS_METERS: float = 50000
    
    # Time windows (24-hour format)
    CHECKIN_START: str = "00:00"
    CHECKIN_END: str = "23:59"
    CHECKOUT_START: str = "00:00"
    CHECKOUT_END: str = "23:59"
    
    # Face recognition settings
    ENCODING_SAMPLES: int = 3
    FACE_MATCH_TOLERANCE: float = 0.5
    ALLOW_ANYTIME_ATTENDANCE: bool = True
    
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_RESIZE_SCALE: float = 0.25


config = Config()
