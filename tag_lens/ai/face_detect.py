from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

# model_selection=1: 카메라에서 5m 이내 원거리/작은 얼굴까지 잡는 full-range 모델
# (0번 short-range 모델은 2m 이내 정면 위주라 Haar Cascade와 큰 차이가 없음)
_FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    mediapipe BlazeFace 기반 얼굴 감지.
    Haar Cascade와 달리 측면/기울어진 얼굴, 저조도, 작은 얼굴도 상당수 검출한다.
    Returns: [(x, y, w, h), ...] 형식
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _FACE_DETECTOR.process(rgb)

    faces: list[tuple[int, int, int, int]] = []
    if not results.detections:
        return faces

    for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        x = max(0, int(box.xmin * w))
        y = max(0, int(box.ymin * h))
        bw = min(w - x, int(box.width * w))
        bh = min(h - y, int(box.height * h))
        if bw > 0 and bh > 0:
            faces.append((x, y, bw, bh))
    return faces


def count_faces(image: np.ndarray) -> int:
    return len(detect_faces(image))


def draw_face_boxes(image: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
    annotated = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated
