# utils.py

import os
import csv
import pickle
from datetime import datetime, time
from typing import Dict, List, Optional, Any
import numpy as np

from config import config


def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    os.makedirs("data", exist_ok=True)


def load_students() -> Dict[str, Dict[str, str]]:
    """Load student metadata from CSV into a dictionary keyed by ERP."""
    students = {}
    if not os.path.exists(config.STUDENTS_FILE):
        return students
    
    fieldnames = ["erp", "name", "branch", "semester", "section"]
    with open(config.STUDENTS_FILE, "r", newline="") as f:
        sample = f.readline()
        if not sample:
            return students
        first_row = next(csv.reader([sample]))
        has_header = [c.strip().lower() for c in first_row] == fieldnames
        f.seek(0)
        if has_header:
            reader = csv.DictReader(f)
        else:
            reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            if not row.get("erp") or not row.get("name"):
                continue
            students[row["erp"]] = {
                "name": row["name"],
                "branch": row.get("branch", ""),
                "semester": row.get("semester", ""),
                "section": row.get("section", "")
            }
    return students


def save_student(erp, name, branch, semester, section):

    students = load_students()

    # Prevent duplicate ERP
    if erp in students:
        return False

    ensure_data_directory()

    file_exists = os.path.exists(config.STUDENTS_FILE)

    fieldnames = [
        "erp",
        "name",
        "branch",
        "semester",
        "section"
    ]

    with open(config.STUDENTS_FILE, "a", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only first time
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "erp": erp,
            "name": name,
            "branch": branch,
            "semester": semester,
            "section": section
        })

    return True


def load_encodings() -> Dict[str, List[np.ndarray]]:
    """Load face encodings from pickle file."""
    if not os.path.exists(config.ENCODINGS_FILE):
        return {}
    
    with open(config.ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_encodings(encodings: Dict[str, List[np.ndarray]]):
    """Save face encodings to pickle file."""
    ensure_data_directory()
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)


def add_encoding(erp: str, encoding: np.ndarray):
    """Add a face encoding for a student."""
    encodings = load_encodings()
    if erp not in encodings:
        encodings[erp] = []
    encodings[erp].append(encoding)
    save_encodings(encodings)


def parse_time(time_str: str) -> time:
    """Parse a time string (HH:MM) into a time object."""
    return datetime.strptime(time_str, "%H:%M").time()


def is_within_time_window(start: str, end: str) -> bool:
    """Check if current time is within the specified window."""
    now = datetime.now().time()
    start_time = parse_time(start)
    end_time = parse_time(end)
    return start_time <= now <= end_time


def get_today_date() -> str:
    """Get today's date as YYYY-MM-DD string."""
    return datetime.now().strftime("%Y-%m-%d")
