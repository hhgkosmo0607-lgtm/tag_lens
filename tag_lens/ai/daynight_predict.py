from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from opencv.preprocess import prepare_for_cnn

CLASS_NAMES = ["주간", "야간"]


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.percentile(gray, 50))
    night_score = max(0.0, min(1.0, (120 - brightness) / 120))
    probs = {"주간": 1.0 - night_score, "야간": night_score}
    label = max(probs, key=probs.get)
    return label, probs


def predict_daynight(image, model_path: str | None = None) -> tuple[str, dict[str, float]]:
    if model_path is None:
        model_path = str(Path("models") / "daynight_model.h5")

    model_file = Path(model_path)
    if not model_file.exists():
        return _heuristic_predict(image)

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        batch = prepare_for_cnn(image)
        pred = model.predict(batch, verbose=0)[0]

        probs = {name: float(pred[idx]) for idx, name in enumerate(CLASS_NAMES)}
        label = CLASS_NAMES[int(np.argmax(pred))]
        return label, probs
    except Exception:
        return _heuristic_predict(image)
