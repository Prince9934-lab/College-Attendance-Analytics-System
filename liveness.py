# liveness.py

import cv2
import numpy as np
from typing import Tuple, Optional
import time


class LivenessDetector:
    """
    Basic liveness detection using blink detection.
    Uses eye aspect ratio (EAR) to detect blinks.
    """
    
    def __init__(self):
        # Load face and eye cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.blink_threshold = 3
        self.blink_count = 0
        self.eyes_closed_frames = 0
        self.eyes_were_closed = False
    
    def detect_eyes(self, frame: np.ndarray) -> int:
        """Detect number of eyes in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        total_eyes = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            total_eyes += len(eyes)
        
        return total_eyes
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Process a frame for blink detection.
        
        Returns:
            (blink_detected_this_frame, total_blink_count)
        """
        eye_count = self.detect_eyes(frame)
        blink_detected = False
        
        if eye_count < 2:
            self.eyes_closed_frames += 1
            if self.eyes_closed_frames > 2:
                self.eyes_were_closed = True
        else:
            if self.eyes_were_closed:
                self.blink_count += 1
                blink_detected = True
            self.eyes_closed_frames = 0
            self.eyes_were_closed = False
        
        return blink_detected, self.blink_count
    
    def is_live(self) -> bool:
        """Check if enough blinks have been detected."""
        return self.blink_count >= self.blink_threshold
    
    def reset(self):
        """Reset the detector state."""
        self.blink_count = 0
        self.eyes_closed_frames = 0
        self.eyes_were_closed = False


def perform_liveness_check(timeout_seconds: int = 10) -> Tuple[bool, str]:
    """
    Perform a liveness check requiring the user to blink.
    
    Returns:
        (passed, message)
    """
    detector = LivenessDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return False, "Could not open camera"
    
    start_time = time.time()
    required_blinks = detector.blink_threshold
    
    print(f"\nLiveness Check: Please blink {required_blinks} times")
    
    try:
        while time.time() - start_time < timeout_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            
            blink_detected, total_blinks = detector.process_frame(frame)
            
            # Draw status
            remaining = required_blinks - total_blinks
            status = f"Blinks: {total_blinks}/{required_blinks}"
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            time_left = int(timeout_seconds - (time.time() - start_time))
            cv2.putText(frame, f"Time: {time_left}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Liveness Check", frame)
            
            if detector.is_live():
                cap.release()
                cv2.destroyAllWindows()
                return True, "Liveness check passed"
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return False, f"Liveness check failed (detected {detector.blink_count} blinks)"

