from __future__ import annotations

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return image


def prepare_for_cnn(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)
