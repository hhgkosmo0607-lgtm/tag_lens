"""
유사 사진 검색용 특징 벡터 추출 — 순수 ImageNet 사전학습 MobileNetV2를 그대로 쓴다.
같은 폴더의 genre_predict.py/daynight_predict.py/indoor_predict.py와 달리
우리가 학습/fine-tuning한 모델이 아니다.

장르 분류기(genre_model.h5)의 레이어를 재사용하는 방식을 예전에 썼었는데,
fine-tuning 과정에서 "5개 장르 구분"에만 최적화되면서 같은 장르 안의 시각적
차이가 뭉개져 유사도 검색이 사실상 "같은 장르인가"로 수렴해버리는 문제가 있었다
(같은 장르 내 무관한 사진끼리도 코사인 유사도 0.9+). 그래서 장르로 fine-tuning되지
않은 순수 사전학습 모델을 특징 추출 전용으로 쓴다.
"""

from __future__ import annotations

import numpy as np

_FEATURE_EXTRACTOR = None


def _get_feature_extractor():
    """특징 추출기(전역 캐시). 최초 1회만 로드하고 이후 재사용한다."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        from tensorflow.keras.applications import MobileNetV2

        _FEATURE_EXTRACTOR = MobileNetV2(
            weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
        )
    return _FEATURE_EXTRACTOR


def extract_features(image) -> np.ndarray:
    """유사 사진 검색용 특징 벡터 추출 (1280차원, L2 정규화)"""
    try:
        import cv2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        extractor = _get_feature_extractor()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224)).astype("float32")
        batch = preprocess_input(np.expand_dims(resized, axis=0))
        features = extractor.predict(batch, verbose=0)[0]
        features = features / (np.linalg.norm(features) + 1e-8)
        return features.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to extract features: {type(e).__name__}: {e}")
        return np.zeros(1280, dtype=np.float32)
