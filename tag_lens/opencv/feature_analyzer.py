from __future__ import annotations

import cv2
import numpy as np


def analyze_features(image: np.ndarray) -> dict[str, object]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 밝기 (Brightness) 분석 ---
    brightness_percentile = float(np.percentile(gray, 50))  # 중앙값
    brightness_label = "밝은" if brightness_percentile >= 120 else "어두운"

    # --- 색온도 (Color Tone) BGR 히스토그램 분석 ---
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
    
    # 가중 평균으로 색온도 판단 (빨강 vs 파랑)
    red_weight = np.sum(hist_r * np.arange(256)) / np.sum(hist_r) if np.sum(hist_r) > 0 else 128
    blue_weight = np.sum(hist_b * np.arange(256)) / np.sum(hist_b) if np.sum(hist_b) > 0 else 128
    color_tone = "따뜻한" if red_weight >= blue_weight else "차가운"

    # --- 채도 (Saturation) HSV 분석 ---
    saturation_percentile = float(np.percentile(hsv[:, :, 1], 75))  # 75 백분위
    saturation_label = "고채도" if saturation_percentile >= 100 else "저채도"

    # --- 흑백 여부 판단 ---
    channel_std = float(np.mean(np.std(image.astype("float32"), axis=2)))
    is_bw = channel_std < 8.0

    return {
        "brightness": brightness_percentile,
        "brightness_label": brightness_label,
        "color_tone": color_tone,
        "saturation": saturation_percentile,
        "saturation_label": saturation_label,
        "is_bw": is_bw,
    }
