from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from opencv.preprocess import prepare_for_cnn

CLASS_NAMES = ["실내", "실외"]

# genre_predict.py의 LOW_CONFIDENCE_THRESHOLD와 같은 이유 — 51% vs 49%처럼 애매한
# 경우까지 억지로 실내/실외를 확정하지 않고, 확신이 낮으면 판별 자체를 건너뛴다.
LOW_CONFIDENCE_THRESHOLD = 0.6


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    """모델이 없을 때의 약한 대체 판별: 사진 상단 1/3에 하늘색(밝고 파란) 영역이 많으면 실외로 추정."""
    h, w = image.shape[:2]
    top_third = image[: h // 3, :, :]
    b, g, r = np.mean(top_third.reshape(-1, 3), axis=0)
    sky_score = max(0.0, min(1.0, ((b - r) / 255.0) + (float(np.mean(top_third)) / 255.0) - 0.5))
    probs = {"실외": sky_score, "실내": 1.0 - sky_score}
    label = max(probs, key=probs.get)
    return label, probs


_MODEL_CACHE: dict[str, object] = {}


def _load_model_cached(model_path: str):
    """모델을 최초 1회만 디스크에서 로드하고 이후 호출은 캐시를 재사용한다.
    매 업로드마다 .h5를 새로 로드하면 요청당 1~3초씩 낭비된다."""
    if model_path not in _MODEL_CACHE:
        import tensorflow as tf

        _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE[model_path]


def predict_indoor(image, model_path: str | None = None) -> tuple[str | None, dict[str, float]]:
    """실내/실외를 판별한다. 확신도가 LOW_CONFIDENCE_THRESHOLD 미만이면 판별을 건너뛰고
    (None, probs)를 반환한다 — tagging.py는 indoor가 None이면 태그를 붙이지 않는다."""
    if model_path is None:
        model_path = str(Path("models") / "indoor_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        label, probs = _heuristic_predict(image)
    else:
        try:
            model = _load_model_cached(model_path)
            batch = prepare_for_cnn(image)
            pred = model.predict(batch, verbose=0)[0]

            probs = {name: float(pred[idx]) for idx, name in enumerate(CLASS_NAMES)}
            label = CLASS_NAMES[int(np.argmax(pred))]
        except Exception:
            label, probs = _heuristic_predict(image)

    if probs[label] < LOW_CONFIDENCE_THRESHOLD:
        return None, probs
    return label, probs
