# registration.py

import os
import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple

from config import config
from utils import save_student, load_students, add_encoding, load_encodings


class RegistrationError(Exception):
    """Custom exception for registration errors."""
    pass


def capture_face_encoding(frame: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Detect and encode a single face from a frame.

    Returns:
        (encoding, message)
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return None, "No face detected"

    if len(face_locations) > 1:
        return None, f"Multiple faces detected ({len(face_locations)}). Only one face allowed."

    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not encodings:
        return None, "Could not generate face encoding"

    return encodings[0], "Face captured successfully"


def draw_face_rectangle(frame: np.ndarray, color=(0, 255, 0)):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    return frame, len(face_locations)


def save_face_image(frame, erp, sample_number):
    """
    Save captured face image.
    """

    folder_path = f"data/faces/{erp}"

    os.makedirs(folder_path, exist_ok=True)

    image_path = f"{folder_path}/sample_{sample_number}.jpg"

    cv2.imwrite(image_path, frame)

    return image_path


def register_student(
    erp: str,
    name: str,
    branch: str,
    semester: str,
    section: str
):

    # Duplicate ERP check
    if erp in load_students():
        return False, f"ERP {erp} already registered"

    if erp in load_encodings():
        return False, f"Face encodings already exist for ERP {erp}"

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        return False, "Could not open camera"

    encodings_captured = 0
    target_encodings = config.ENCODING_SAMPLES

    print("\n====================================")
    print(f"Registering Student: {name}")
    print(f"ERP: {erp}")
    print("====================================")
    print(f"Need {target_encodings} face samples")
    print("Press 'c' to capture")
    print("Press 'q' to quit\n")

    try:

        while encodings_captured < target_encodings:

            ret, frame = cap.read()

            if not ret:
                continue

            display_frame, face_count = draw_face_rectangle(frame.copy())

            status = f"Captured: {encodings_captured}/{target_encodings}"

            cv2.putText(
                display_frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if face_count != 1:

                cv2.putText(
                    display_frame,
                    "Only one face allowed",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Student Registration", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q'):
                return False, "Registration cancelled"

            # Capture
            if key == ord('c'):

                encoding, message = capture_face_encoding(frame)

                if encoding is not None:

                    # Save encoding
                    add_encoding(erp, encoding)

                    # Save image
                    image_path = save_face_image(
                        frame,
                        erp,
                        encodings_captured + 1
                    )

                    encodings_captured += 1

                    print(f"\nSample {encodings_captured} captured")
                    print(f"Saved image: {image_path}")

                else:
                    print(f"\nCapture Failed: {message}")

    finally:

        cap.release()
        cv2.destroyAllWindows()

    if encodings_captured < target_encodings:
        return False, "Not enough face samples captured"

    # Save student details
    saved = save_student(
        erp,
        name,
        branch,
        semester,
        section
    )

    if not saved:
        return False, "Failed to save student data"

    return True, f"Successfully registered {name}"


def interactive_registration():

    print("\n" + "=" * 50)
    print("STUDENT REGISTRATION")
    print("=" * 50)

    erp = input("Enter ERP: ").strip()
    name = input("Enter Name: ").strip()
    branch = input("Enter Branch: ").strip()
    semester = input("Enter Semester: ").strip()
    section = input("Enter Section: ").strip()

    if not all([erp, name, branch, semester, section]):
        print("All fields are required")
        return

    success, message = register_student(
        erp,
        name,
        branch,
        semester,
        section
    )

    if success:
        print(f"\nSUCCESS: {message}")
    else:
        print(f"\nFAILED: {message}")