from __future__ import annotations

from pathlib import Path

import numpy as np

from opencv.preprocess import prepare_for_cnn

CLASS_NAMES = ["풍경", "도시", "음식", "사물"]


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    h, w, _ = image.shape
    b, g, r = np.mean(image.reshape(-1, 3), axis=0)

    scores = {
        "풍경": 0.25,
        "도시": 0.25,
        "음식": 0.25,
        "사물": 0.25,
    }

    if abs(h - w) < min(h, w) * 0.2:
        scores["도시"] += 0.08
    if r > b and r > g:
        scores["음식"] += 0.10
    if b > r:
        scores["풍경"] += 0.08

    total = sum(scores.values())
    probs = {k: float(v / total) for k, v in scores.items()}
    genre = max(probs, key=probs.get)
    return genre, probs


def predict_genre(image, model_path: str | None = None) -> tuple[str, dict[str, float]]:
    if model_path is None:
        model_path = str(Path("models") / "genre_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        return _heuristic_predict(image)

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        batch = prepare_for_cnn(image)
        pred = model.predict(batch, verbose=0)[0]

        probs = {name: float(pred[idx]) for idx, name in enumerate(CLASS_NAMES)}
        genre = CLASS_NAMES[int(np.argmax(pred))]
        return genre, probs
    except Exception:
        return _heuristic_predict(image)


def extract_features(image, model_path: str | None = None) -> np.ndarray:
    """CNN의 중간 레이어에서 특징 벡터 추출 (유사도 검색용)"""
    if model_path is None:
        model_path = str(Path("models") / "genre_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[DEBUG] Model not found at {model_path}, returning zero vector")
        return np.zeros(1024, dtype=np.float32)

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        
        # 모델 출력 레이어 수 확인 (3-class vs 4-class 감지)
        output_shape = model.output.shape
        num_classes = output_shape[-1]
        print(f"[DEBUG] Model output shape: {output_shape} (num_classes: {num_classes})")
        
        # 마지막 Dense 레이어 전의 특징 추출 (분류 전 특징 벡터)
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output  # 마지막 Dense 이전
        )
        batch = prepare_for_cnn(image)
        features = feature_model.predict(batch, verbose=0)[0]
        # L2 정규화
        features = features / (np.linalg.norm(features) + 1e-8)
        return features.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to extract features: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(1024, dtype=np.float32)
    except Exception:
        return np.zeros(1024, dtype=np.float32)
