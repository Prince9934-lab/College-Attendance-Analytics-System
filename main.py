# main.py

import cv2
import sys
from datetime import datetime
from typing import Tuple

from config import config
from registration import register_student, interactive_registration
from recognition import FaceRecognizer
from location import LocationValidator
from attendance import AttendanceManager, AttendanceAction
from liveness import perform_liveness_check
from utils import load_students, get_today_date


class AttendanceSystem:
    """Main attendance system integrating all components."""
    
    def __init__(self, enable_liveness: bool = False):
        self.recognizer = FaceRecognizer()
        self.location_validator = LocationValidator()
        self.attendance_manager = AttendanceManager()
        self.enable_liveness = enable_liveness
    
    def validate_all_conditions(self) -> Tuple[bool, str]:
        """Validate location conditions and allow attendance any time."""
        time_msg = "Any-time attendance mode enabled"
        
        # Check location
        loc_valid, loc_msg, distance = self.location_validator.validate()
        if not loc_valid:
            return False, f"Location validation failed: {loc_msg}"
        
        return True, f"All conditions valid. {time_msg}"
    
    def process_attendance(self, frame) -> Tuple[bool, str, dict]:
        """
        Complete attendance processing pipeline.
        
        Returns:
            (success, message, details_dict)
        """
        details = {
            "erp": None,
            "name": None,
            "action": None,
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": get_today_date()
        }
        
        # Step 1: Validate conditions
        conditions_valid, conditions_msg = self.validate_all_conditions()
        if not conditions_valid:
            return False, conditions_msg, details
        
        # Step 2: Recognize face
        erp, student_info, recog_msg = self.recognizer.recognize_face(frame)
        if erp is None:
            return False, f"Face recognition failed: {recog_msg}", details
        
        details["erp"] = erp
        details["name"] = student_info.get("name", "Unknown")
        
        # Step 3: Record attendance
        success, attend_msg = self.attendance_manager.process_attendance(erp)
        
        action, _ = self.attendance_manager.determine_action(erp)
        details["action"] = action.value if action else None
        
        if success:
            return True, f"{details['name']}: {attend_msg}", details
        else:
            return False, f"{details['name']}: {attend_msg}", details
    
    def run_live_attendance(self):
        """Run the attendance system with live camera feed."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("ATTENDANCE SYSTEM - LIVE MODE")
        print("="*50)
        print("Press 'a' to mark attendance")
        print("Press 'l' to perform liveness check (if enabled)")
        print("Press 'r' to reload data")
        print("Press 'q' to quit\n")
        
        # Initial condition check
        valid, msg = self.validate_all_conditions()
        print(f"System Status: {msg}\n")
        
        liveness_passed = not self.enable_liveness  # Skip if not enabled
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Draw face rectangles
                locations = self.recognizer.get_face_locations_scaled(frame)
                for top, right, bottom, left in locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Show status
                action, time_msg = self.attendance_manager.determine_action()
                action_text = action.value.upper() if action else "CLOSED"
                
                cv2.putText(frame, f"Mode: {action_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces: {len(locations)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.enable_liveness:
                    liveness_text = "PASSED" if liveness_passed else "REQUIRED"
                    color = (0, 255, 0) if liveness_passed else (0, 0, 255)
                    cv2.putText(frame, f"Liveness: {liveness_text}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Attendance System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                if key == ord('r'):
                    self.recognizer.reload_data()
                    print("Data reloaded")
                
                if key == ord('l') and self.enable_liveness:
                    cv2.destroyAllWindows()
                    passed, msg = perform_liveness_check()
                    liveness_passed = passed
                    print(f"Liveness: {msg}")
                    cap = cv2.VideoCapture(config.CAMERA_INDEX)
                
                if key == ord('a'):
                    if self.enable_liveness and not liveness_passed:
                        print("Please complete liveness check first (press 'l')")
                        continue
                    
                    success, message, details = self.process_attendance(frame)
                    status = "SUCCESS" if success else "FAILED"
                    print(f"[{details['time']}] {status}: {message}")
                    
                    # Reset liveness for next person
                    if self.enable_liveness:
                        liveness_passed = False
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def show_menu():
    """Display the main menu."""
    print("\n" + "="*50)
    print("FACIAL RECOGNITION ATTENDANCE SYSTEM")
    print("="*50)
    print("1. Register new student")
    print("2. Start attendance (live mode)")
    print("3. Start attendance with liveness detection")
    print("4. View today's attendance")
    print("5. Finalize day's attendance")
    print("6. Test face recognition")
    print("7. Check system status")
    print("8. Toggle any-time attendance mode")
    print("0. Exit")
    print("="*50)


def view_attendance():
    """Display today's attendance report."""
    manager = AttendanceManager()
    students = load_students()
    report = manager.get_attendance_report()
    
    print("\n" + "-"*60)
    print(f"ATTENDANCE REPORT - {get_today_date()}")
    print("-"*60)
    
    if not report:
        print("No attendance records for today")
    else:
        print(f"{'ERP':<12} {'Name':<20} {'Check-In':<12} {'Check-Out':<12} {'Status'}")
        print("-"*60)
        for record in report:
            name = students.get(record['erp'], {}).get('name', 'Unknown')[:18]
            print(f"{record['erp']:<12} {name:<20} {record['checkin']:<12} {record['checkout']:<12} {record['status']}")
    
    print("-"*60)


def check_system_status():
    """Check and display system status."""
    from location import validate_location
    
    print("\n" + "-"*40)
    print("SYSTEM STATUS")
    print("-"*40)
    
    # Time status
    manager = AttendanceManager()
    action, time_msg = manager.determine_action()
    print(f"Time Window: {time_msg}")
    print(f"  Check-in:  {config.CHECKIN_START} - {config.CHECKIN_END}")
    print(f"  Check-out: {config.CHECKOUT_START} - {config.CHECKOUT_END}")
    print(f"Any-time attendance mode: {'ENABLED' if config.ALLOW_ANYTIME_ATTENDANCE else 'DISABLED'}")
    
    # Location status
    loc_valid, loc_msg, _ = validate_location()
    print(f"\nLocation: {loc_msg}")
    print(f"  Campus: {config.CAMPUS_LOCATION}")
    print(f"  Radius: {config.ALLOWED_RADIUS_METERS}m")
    
    # Data status
    students = load_students()
    from utils import load_encodings
    encodings = load_encodings()
    print(f"\nData:")
    print(f"  Registered students: {len(students)}")
    print(f"  Face encodings: {sum(len(e) for e in encodings.values())}")
    
    print("-"*40)


def main():
    """Main entry point."""
    while True:
        show_menu()
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            interactive_registration()
        
        elif choice == "2":
            system = AttendanceSystem(enable_liveness=False)
            system.run_live_attendance()
        
        elif choice == "3":
            system = AttendanceSystem(enable_liveness=True)
            system.run_live_attendance()
        
        elif choice == "4":
            view_attendance()
        
        elif choice == "5":
            manager = AttendanceManager()
            updated = manager.finalize_day()
            if updated:
                print("Day's attendance finalized")
            else:
                print("No records to finalize")
        
        elif choice == "6":
            from recognition import test_recognition
            test_recognition()
        
        elif choice == "7":
            check_system_status()
        
        elif choice == "8":
            toggle_anytime_attendance()
        
        elif choice == "0":
            print("Exiting...")
            sys.exit(0)
        
        else:
            print("Invalid choice")


def toggle_anytime_attendance():
    """Toggle whether attendance is allowed at any time."""
    current = config.ALLOW_ANYTIME_ATTENDANCE
    print(f"\nAny-time attendance mode is currently {'ENABLED' if current else 'DISABLED'}.")
    choice = input("Would you like to toggle it? (y/n): ").strip().lower()
    if choice == 'y':
        config.ALLOW_ANYTIME_ATTENDANCE = not current
        print(f"Any-time attendance mode is now {'ENABLED' if config.ALLOW_ANYTIME_ATTENDANCE else 'DISABLED'}.")
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
