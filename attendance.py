# attendance.py

import csv
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from enum import Enum

from config import config
from utils import get_today_date, is_within_time_window, ensure_data_directory


class AttendanceAction(Enum):
    CHECKIN = "checkin"
    CHECKOUT = "checkout"


class AttendanceRecord:
    """Represents a single attendance record."""
    
    def __init__(
        self,
        erp: str,
        date: str,
        checkin: str = "",
        checkout: str = "",
        status: str = "PENDING"
    ):
        self.erp = erp
        self.date = date
        self.checkin = checkin
        self.checkout = checkout
        self.status = status
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "erp": self.erp,
            "date": self.date,
            "checkin": self.checkin,
            "checkout": self.checkout,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "AttendanceRecord":
        return cls(
            erp=data["erp"],
            date=data["date"],
            checkin=data.get("checkin", ""),
            checkout=data.get("checkout", ""),
            status=data.get("status", "PENDING")
        )


class AttendanceManager:
    """Manages attendance records and validation."""
    
    FIELDNAMES = ["erp", "date", "checkin", "checkout", "status"]
    
    def __init__(self):
        ensure_data_directory()
    
    def _load_all_records(self) -> List[AttendanceRecord]:
        """Load all attendance records from CSV."""
        records = []
        if not os.path.exists(config.ATTENDANCE_FILE):
            return records
        
        with open(config.ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(AttendanceRecord.from_dict(row))
        
        return records
    
    def _save_all_records(self, records: List[AttendanceRecord]):
        """Save all attendance records to CSV."""
        with open(config.ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())
    
    def get_record(self, erp: str, date: str) -> Optional[AttendanceRecord]:
        """Get attendance record for a student on a specific date."""
        records = self._load_all_records()
        for record in records:
            if record.erp == erp and record.date == date:
                return record
        return None
    
    def get_today_record(self, erp: str) -> Optional[AttendanceRecord]:
        """Get today's attendance record for a student."""
        return self.get_record(erp, get_today_date())
    
    def determine_action(self, erp: Optional[str] = None) -> Tuple[Optional[AttendanceAction], str]:
        """Determine which action is valid based on current time or student state."""
        if is_within_time_window(config.CHECKIN_START, config.CHECKIN_END):
            return AttendanceAction.CHECKIN, "Check-in window is open"
        
        if is_within_time_window(config.CHECKOUT_START, config.CHECKOUT_END):
            return AttendanceAction.CHECKOUT, "Check-out window is open"
        
        if config.ALLOW_ANYTIME_ATTENDANCE:
            if erp is not None:
                existing = self.get_today_record(erp)
                if not existing or not existing.checkin:
                    return AttendanceAction.CHECKIN, "Any-time attendance allowed: check-in"
                if existing.checkin and not existing.checkout:
                    return AttendanceAction.CHECKOUT, "Any-time attendance allowed: check-out"
                return None, "Attendance already completed for today"
            return AttendanceAction.CHECKIN, "Any-time attendance allowed"
        
        now = datetime.now().strftime("%H:%M")
        return None, f"Outside attendance windows (current time: {now})"
    
    def record_checkin(self, erp: str) -> Tuple[bool, str]:
        """Record a check-in for a student."""
        today = get_today_date()
        existing = self.get_today_record(erp)
        
        if existing and existing.checkin:
            return False, f"Already checked in at {existing.checkin}"
        
        now_time = datetime.now().strftime("%H:%M:%S")
        
        records = self._load_all_records()
        
        if existing:
            # Update existing record
            for record in records:
                if record.erp == erp and record.date == today:
                    record.checkin = now_time
                    break
        else:
            # Create new record
            records.append(AttendanceRecord(
                erp=erp,
                date=today,
                checkin=now_time
            ))
        
        self._save_all_records(records)
        return True, f"Check-in recorded at {now_time}"
    
    def record_checkout(self, erp: str) -> Tuple[bool, str]:
        """Record a check-out for a student."""
        today = get_today_date()
        existing = self.get_today_record(erp)
        
        if not existing or not existing.checkin:
            return False, "Cannot check out without checking in first"
        
        if existing.checkout:
            return False, f"Already checked out at {existing.checkout}"
        
        now_time = datetime.now().strftime("%H:%M:%S")
        
        records = self._load_all_records()
        for record in records:
            if record.erp == erp and record.date == today:
                record.checkout = now_time
                record.status = "PRESENT"
                break
        
        self._save_all_records(records)
        return True, f"Check-out recorded at {now_time}. Status: PRESENT"
    
    def process_attendance(self, erp: str) -> Tuple[bool, str]:
        """
        Process attendance based on current time window or any-time mode.
        Automatically determines whether to check-in or check-out.
        """
        action, message = self.determine_action(erp)
        
        if action is None:
            return False, message
        
        if action == AttendanceAction.CHECKIN:
            return self.record_checkin(erp)
        else:
            return self.record_checkout(erp)
    
    def finalize_day(self, date: Optional[str] = None):
        """
        Finalize attendance for a day.
        Mark incomplete records as ABSENT.
        """
        if date is None:
            date = get_today_date()
        
        records = self._load_all_records()
        updated = False
        
        for record in records:
            if record.date == date and record.status == "PENDING":
                if record.checkin and record.checkout:
                    record.status = "PRESENT"
                else:
                    record.status = "ABSENT"
                updated = True
        
        if updated:
            self._save_all_records(records)
        
        return updated
    
    def get_attendance_report(self, date: Optional[str] = None) -> List[Dict]:
        """Get attendance report for a specific date."""
        if date is None:
            date = get_today_date()
        
        records = self._load_all_records()
        return [
            record.to_dict()
            for record in records
            if record.date == date
        ]
