from __future__ import annotations

import cv2
import numpy as np


def create_average_hash(image, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(np.mean(resized))
    bits = resized >= avg
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if len(hash_a) != len(hash_b):
        raise ValueError("두 해시 길이가 다릅니다.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash_a, hash_b))
