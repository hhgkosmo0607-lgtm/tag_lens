"""
얼굴 검출 — mediapipe(Google)의 사전학습 BlazeFace 모델을 그대로 사용한다.
같은 폴더의 genre_predict.py/daynight_predict.py/indoor_predict.py와 달리
우리가 직접 학습시킨 모델이 아니라 외부에서 가져온 사전학습 모델이다.

2026-07-09: BlazeFace가 원형 대칭 물체(벽에 붙은 화재경보기 벨)를 얼굴로 오탐하는
사례 발견(신뢰도 0.696, min_detection_confidence 임계값 통과). 검출 신뢰도나 얼굴
특징점(눈/코/입 좌표) 기하학으로는 진짜 얼굴과 깔끔하게 안 갈림 — 오탐 신뢰도(0.696)가
일부 진짜 얼굴(0.656~0.691)보다 오히려 높고, 특징점 비율도 겹침. 대신 CLIP 제로샷으로
2차 검증을 추가: 검출된 박스를 60% 여백을 두고 잘라내서 "사람 얼굴 클로즈업" vs "얼굴이
아닌 원형 물체/표지판/버튼"을 물어봄 — 실측(진짜 얼굴 10건 vs 오탐 1건)에서 진짜 얼굴은
전부 0.270 이상, 오탐은 0.049로 깨끗하게 갈림(_FACE_CROP_VERIFY_THRESHOLD=0.15로 그
사이에 설정). 부수 효과로 "실제로는 얼굴이지만 화면 대비 너무 작아서 크롭해도 얼굴처럼
안 보이는" 배경 속 행인 얼굴(0.031)도 같이 걸러짐 — 이건 오탐 방지와는 별개로, 사진의
주제가 아닌 배경 속 얼굴 때문에 장르가 통째로 "인물"로 강제되는 문제도 완화해줌.
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
# min_detection_confidence=0.65: full-range 모델은 재현율을 높이려고 오탐이 잦은데,
# 노을 진 구름(신뢰도 0.53~0.54)에 이어 2026-07-09에 실사용 중 새 오탐 발견 —
# 벽에 붙은 사진 콜라주 속 자판기 버튼 패널(원형+대칭 격자 무늬)이 0.642로 잡힘.
# 실측(실제 업로드 82장): 이 오탐(0.642)과 가장 가까운 진짜 얼굴(0.656, 인물 사진 확인됨)
# 사이가 0.014밖에 안 나서, 그 중간인 0.65로 올려 둘을 가른다. 0.6→0.65로는 82장 중
# 진짜 얼굴 검출 결과에 영향 없음을 확인(가장 낮은 진짜 얼굴도 0.656으로 여전히 통과).
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_full_range.tflite"
_base_options = mp_tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
_options = mp_vision.FaceDetectorOptions(base_options=_base_options, min_detection_confidence=0.65)
_FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(_options)

# 검출된 박스 주변 여백 비율(가로/세로 각각 60%) — 너무 딱 맞게 자르면 얼굴 여부를
# 판단할 맥락(머리카락, 목, 주변 윤곽)이 없어져 CLIP도 판단하기 어려워진다.
_CROP_MARGIN_RATIO = 0.6
_FACE_CROP_PROMPTS = [
    "a close-up photo of a human face",
    "a photo of a round object, sign, button, or device that is not a face",
]
_FACE_CROP_VERIFY_THRESHOLD = 0.15


def _verify_face_crop(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """검출된 얼굴 박스를 잘라 CLIP으로 2차 검증한다. 실패 시(모델 로드 불가 등) 그냥 통과시킨다."""
    try:
        import torch

        from ai.clip_utils import get_clip, image_to_pil

        h, w = image.shape[:2]
        x, y, bw, bh = box
        mx, my = int(bw * _CROP_MARGIN_RATIO), int(bh * _CROP_MARGIN_RATIO)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        crop = image[y0:y1, x0:x1]

        model, processor = get_clip()
        inputs = processor(text=_FACE_CROP_PROMPTS, images=image_to_pil(crop), return_tensors="pt", padding=True)
        with torch.no_grad():
            scores = model(**inputs).logits_per_image.softmax(dim=1)[0]
        return float(scores[0]) >= _FACE_CROP_VERIFY_THRESHOLD
    except Exception as e:
        print(f"[ERROR] Failed to verify face crop via CLIP: {type(e).__name__}: {e}")
        return True


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    mediapipe BlazeFace(Tasks API) 기반 얼굴 감지 + CLIP 2차 검증.
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
        if bw > 0 and bh > 0 and _verify_face_crop(image, (x, y, bw, bh)):
            faces.append((x, y, bw, bh))
    return faces


def count_faces(image: np.ndarray) -> int:
    return len(detect_faces(image))


def draw_face_boxes(image: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
    annotated = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated
