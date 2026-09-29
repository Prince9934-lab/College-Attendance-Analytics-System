# recognition.py

import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple, Dict, List

from config import config
from utils import load_encodings, load_students


class FaceRecognizer:
    """Real-time face recognition system."""

    def __init__(self):

        self.known_encodings: Dict[str, List[np.ndarray]] = {}
        self.students: Dict[str, Dict[str, str]] = {}

        self._flat_encodings: List[np.ndarray] = []
        self._encoding_erps: List[str] = []

        self.reload_data()

    def reload_data(self):
        """Reload encodings and student data from files."""

        self.known_encodings = load_encodings()
        self.students = load_students()

        # Flatten all encodings
        self._flat_encodings = []
        self._encoding_erps = []

        for erp, encodings in self.known_encodings.items():

            for enc in encodings:

                self._flat_encodings.append(enc)
                self._encoding_erps.append(erp)

        print(
            f"Loaded {len(self.students)} students with "
            f"{len(self._flat_encodings)} face encodings"
        )

    def recognize_face(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[str], Optional[Dict[str, str]], str]:

        """
        Recognize a face from frame
        """

        if not self._flat_encodings:
            return None, None, "No registered faces in system"

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize frame
        small_frame = cv2.resize(
            rgb_frame,
            (0, 0),
            fx=config.FRAME_RESIZE_SCALE,
            fy=config.FRAME_RESIZE_SCALE
        )

        # Detect faces
        face_locations = face_recognition.face_locations(
            small_frame
        )

        if len(face_locations) == 0:
            return None, None, "No face detected"

        if len(face_locations) > 1:
            return None, None, (
                f"Multiple faces detected "
                f"({len(face_locations)})"
            )

        # Encode face
        face_encodings = face_recognition.face_encodings(
            small_frame,
            face_locations
        )

        if not face_encodings:
            return None, None, "Could not encode face"

        face_encoding = face_encodings[0]

        # Compare faces
        distances = face_recognition.face_distance(
            self._flat_encodings,
            face_encoding
        )

        if len(distances) == 0:
            return None, None, "No matches found"

        best_match_idx = np.argmin(distances)

        best_distance = distances[best_match_idx]

        # Match found
        if best_distance <= config.FACE_MATCH_TOLERANCE:

            erp = self._encoding_erps[best_match_idx]

            # Safe student data
            student_info = self.students.get(
                erp,
                {
                    "name": "Unknown",
                    "branch": "",
                    "semester": "",
                    "section": ""
                }
            )

            confidence = (1 - best_distance) * 100

            return (
                erp,
                student_info,
                f"Match found "
                f"(confidence: {confidence:.1f}%)"
            )

        return None, None, (
            f"No match "
            f"(best distance: {best_distance:.2f})"
        )

    def get_face_locations_scaled(
        self,
        frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:

        """
        Get scaled face locations
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        small_frame = cv2.resize(
            rgb_frame,
            (0, 0),
            fx=config.FRAME_RESIZE_SCALE,
            fy=config.FRAME_RESIZE_SCALE
        )

        locations = face_recognition.face_locations(
            small_frame
        )

        scale = 1 / config.FRAME_RESIZE_SCALE

        return [
            (
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale)
            )
            for top, right, bottom, left in locations
        ]


def test_recognition():
    """
    Test live face recognition
    """

    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(
        config.CAMERA_INDEX
    )

    if not cap.isOpened():

        print("Error: Could not open camera")
        return

    print("\nFace Recognition Test")
    print("Press 'r' to recognize")
    print("Press 'q' to quit\n")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            # Draw face boxes
            locations = recognizer.get_face_locations_scaled(
                frame
            )

            for top, right, bottom, left in locations:

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Faces: {len(locations)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Recognition Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q'):
                break

            # Recognize
            if key == ord('r'):

                erp, info, message = recognizer.recognize_face(
                    frame
                )

                if erp and info:

                    print(
                        f"Recognized: "
                        f"{info['name']} "
                        f"(ERP: {erp})"
                    )

                else:

                    print(
                        f"Recognition failed: "
                        f"{message}"
                    )

    finally:

        cap.release()

        cv2.destroyAllWindows()