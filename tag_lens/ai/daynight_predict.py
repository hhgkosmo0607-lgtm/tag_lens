from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from opencv.feature_analyzer import is_bw_image
from opencv.preprocess import prepare_for_cnn

CLASS_NAMES = ["주간", "야간"]

DAY_START_HOUR = 6   # 06:00
DAY_END_HOUR = 18    # 18:00 (미만)


def daynight_from_hour(hour: int) -> tuple[str, dict[str, float]]:
    """EXIF 촬영 시각(0~23시)으로 주간/야간을 확정 판별한다.
    픽셀 밝기를 추정하는 CNN/휴리스틱보다 신뢰도가 훨씬 높은 실측 신호다."""
    label = "주간" if DAY_START_HOUR <= hour < DAY_END_HOUR else "야간"
    return label, {"주간": 1.0 if label == "주간" else 0.0, "야간": 1.0 if label == "야간" else 0.0}


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.percentile(gray, 50))
    night_score = max(0.0, min(1.0, (120 - brightness) / 120))
    probs = {"주간": 1.0 - night_score, "야간": night_score}
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


def predict_daynight(image, model_path: str | None = None) -> tuple[str | None, dict[str, float] | None]:
    """주간/야간을 판별한다. 흑백 사진은 색온도·밝기 신호가 왜곡되므로 판별하지 않고 (None, None)을 반환한다."""
    if is_bw_image(image):
        return None, None

    if model_path is None:
        model_path = str(Path("models") / "daynight_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        return _heuristic_predict(image)

    try:
        model = _load_model_cached(model_path)
        batch = prepare_for_cnn(image)
        pred = model.predict(batch, verbose=0)[0]

        probs = {name: float(pred[idx]) for idx, name in enumerate(CLASS_NAMES)}
        label = CLASS_NAMES[int(np.argmax(pred))]
        return label, probs
    except Exception:
        return _heuristic_predict(image)


def resolve_daynight(
    image, exif_hour: int | None, model_path: str | None = None
) -> tuple[str | None, dict[str, float] | None]:
    """
    주간/야간 최종 판정 — app.py(업로드)와 reclassify_photos.py(재분류)가 공유하는 단일 진입점.
    흑백 사진은 판별하지 않는다(색/밝기 신호가 왜곡되어 (None, None) 반환).

    EXIF 촬영시각이 있으면 그대로 확정 판정(daynight_from_hour), 없으면 CNN(모델
    없으면 밝기 휴리스틱)으로 폴백한다.

    [설계 보류] EXIF·CNN·밝기 휴리스틱 세 신호를 다수결로 종합하는 방식을 시도했었으나,
    CNN과 밝기 휴리스틱이 둘 다 픽셀 밝기에 의존해 독립적이지 않다는 게 실측으로 확인됨
    (그림자 짙은 대낮 숲 사진 250장 중 118장에서 CNN이 야간으로 오판했고, 그중 다수는
    휴리스틱도 똑같이 야간으로 오판 — 이 경우 다수결이 2:1로 정확한 EXIF를 뒤집어버림).
    카메라 시계 오류(EXIF가 실제와 다른 극소수 사례)보다 이 역효과가 더 크다고 판단해
    일단 EXIF 우선 방식으로 되돌려둠 — 재설계는 다음 세션으로 보류(DEVLOG 참고).
    """
    if is_bw_image(image):
        return None, None

    if exif_hour is not None:
        return daynight_from_hour(exif_hour)

    return predict_daynight(image, model_path)
