from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# OpenCV 4.x / 5.x 모두 대응
def _load_cascade() -> cv2.CascadeClassifier | None:
    """Haar Cascade 로드 (여러 경로 시도)"""
    # 시도할 경로 순서
    candidates = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        str(Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"),
    ]
    
    # cv2.CascadeClassifier 위치 탐색 (4.x: cv2, 5.x: cv2.objdetect)
    CascadeClass = getattr(cv2, "CascadeClassifier", None)
    if CascadeClass is None:
        objdetect = getattr(cv2, "objdetect", None)
        if objdetect:
            CascadeClass = getattr(objdetect, "CascadeClassifier", None)

    if CascadeClass is None:
        print("[DEBUG] CascadeClassifier not found in cv2 or cv2.objdetect")
        return None

    for xml_path in candidates:
        try:
            cascade = CascadeClass(xml_path)
            if not cascade.empty():
                print(f"[DEBUG] Successfully loaded cascade from: {xml_path}")
                return cascade
        except Exception as e:
            print(f"[DEBUG] Failed to load from {xml_path}: {e}")
            continue
    
    print(f"[DEBUG] Could not load cascade from any path: {candidates}")
    return None


_FACE_CASCADE = _load_cascade()


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Haar Cascade를 사용한 얼굴 감지 (안정적)
    Returns: [(x, y, w, h), ...] 형식
    """
    if _FACE_CASCADE is None or _FACE_CASCADE.empty():
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 더 세밀하게 탐색
        minNeighbors=4,    # 조금 덜 엄격하게 (놓치는 얼굴 줄이기)
        minSize=(30, 30),
    )
    return [tuple(map(int, face)) for face in faces]


def count_faces(image: np.ndarray) -> int:
    return len(detect_faces(image))


def draw_face_boxes(image: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
    annotated = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated
