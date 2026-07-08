"""
얼굴 검출 — mediapipe(Google)의 사전학습 BlazeFace 모델을 그대로 사용한다.
같은 폴더의 genre_predict.py/daynight_predict.py/indoor_predict.py와 달리
우리가 직접 학습시킨 모델이 아니라 외부에서 가져온 사전학습 모델이다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# 설치된 mediapipe(0.10.x, Python 3.12 Windows 빌드)에는 구버전 Solutions API
# (mp.solutions.face_detection)가 빠져있어서, 신버전 Tasks API로 얼굴을 검출한다.
# blaze_face_full_range: 카메라에서 5m 이내 원거리/작은 얼굴까지 잡는 모델
# (short-range 모델은 2m 이내 정면 위주라 Haar Cascade와 큰 차이가 없음)
#
# min_detection_confidence=0.6: full-range 모델은 재현율을 높이려고 오탐이 잦은데,
# 특히 노을 진 구름처럼 유기적인 텍스처를 얼굴로 착각하는 사례가 실사용 중 확인됨
# (신뢰도 0.53~0.54 수준). 실제 얼굴은 대부분 0.64 이상으로 나와서 0.6으로 올려 오탐을 줄인다.
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_full_range.tflite"
_base_options = mp_tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
_options = mp_vision.FaceDetectorOptions(base_options=_base_options, min_detection_confidence=0.6)
_FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(_options)


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    mediapipe BlazeFace(Tasks API) 기반 얼굴 감지.
    Haar Cascade와 달리 측면/기울어진 얼굴, 저조도, 작은 얼굴도 상당수 검출한다.
    Returns: [(x, y, w, h), ...] 형식
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _FACE_DETECTOR.detect(mp_image)

    faces: list[tuple[int, int, int, int]] = []
    for detection in result.detections:
        box = detection.bounding_box  # Tasks API는 절대 픽셀 좌표로 반환
        x = max(0, box.origin_x)
        y = max(0, box.origin_y)
        bw = min(w - x, box.width)
        bh = min(h - y, box.height)
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
