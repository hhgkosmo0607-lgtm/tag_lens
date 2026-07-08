from __future__ import annotations

from pathlib import Path

import numpy as np

from opencv.preprocess import prepare_for_cnn

CLASS_NAMES = ["자연", "도시", "음식", "사물", "인물"]

# CNN이 5개 장르 중 하나를 억지로 고르지 않고, 확신이 낮으면 "기타"로 분류한다.
# (예: 애매한 클로즈업/아웃포커스 사진이 51%짜리 확률로 엉뚱한 장르에 배정되는 문제)
LOW_CONFIDENCE_THRESHOLD = 0.6


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    h, w, _ = image.shape
    b, g, r = np.mean(image.reshape(-1, 3), axis=0)

    scores = {
        "자연": 0.2,
        "도시": 0.2,
        "음식": 0.2,
        "사물": 0.2,
        "인물": 0.2,
    }

    if abs(h - w) < min(h, w) * 0.2:
        scores["도시"] += 0.08
    if r > b and r > g:
        scores["음식"] += 0.10
    if b > r:
        scores["자연"] += 0.08

    total = sum(scores.values())
    probs = {k: float(v / total) for k, v in scores.items()}
    genre = max(probs, key=probs.get)
    return genre, probs


_MODEL_CACHE: dict[str, object] = {}


def _load_model_cached(model_path: str):
    """모델을 최초 1회만 디스크에서 로드하고 이후 호출은 캐시를 재사용한다.
    매 업로드마다 .h5를 새로 로드하면 요청당 1~3초씩 낭비된다."""
    if model_path not in _MODEL_CACHE:
        import tensorflow as tf

        _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE[model_path]


def predict_genre(image, model_path: str | None = None) -> tuple[str, dict[str, float]]:
    if model_path is None:
        model_path = str(Path("models") / "genre_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        return _heuristic_predict(image)

    try:
        model = _load_model_cached(model_path)
        batch = prepare_for_cnn(image)
        pred = model.predict(batch, verbose=0)[0]

        probs = {name: float(pred[idx]) for idx, name in enumerate(CLASS_NAMES)}
        genre = CLASS_NAMES[int(np.argmax(pred))]
        if probs[genre] < LOW_CONFIDENCE_THRESHOLD:
            genre = "기타"
        return genre, probs
    except Exception:
        return _heuristic_predict(image)


def resolve_genre_with_animal(genre: str, animal: str | None, face_count: int) -> str:
    """
    장르 CNN의 "인물" 학습 데이터가 전부 사람 얼굴 클로즈업(정면 응시, 큰 눈, 얼굴 중심
    구도)이라, 같은 구도의 동물 클로즈업(특히 고양이)을 사람으로 착각해 "인물"로
    오분류하는 사례가 확인됨(실측: 고양이 사진이 96.6% 확신도로 인물 판정). mediapipe가
    사람 얼굴을 못 찾았는데(face_count == 0) animal_predict가 동물을 감지했다면,
    장르 CNN의 "인물" 판정을 신뢰하지 않고 "기타"로 보정한다. "동물"을 별도 장르로
    만들지 않는 이유는 CNN 확률(genre_probs)이 5-class 그대로라 근거 없는 장르를
    새로 만들면 확률 표시와 모순되기 때문 — #고양이 등 동물 태그가 이미 "무엇인지"를
    알려주므로 "#기타 #고양이" 조합으로 충분하다.
    """
    if genre == "인물" and face_count == 0 and animal is not None:
        return "기타"
    return genre
