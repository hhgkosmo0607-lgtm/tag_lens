from __future__ import annotations

import cv2
import numpy as np


def is_bw_image(image: np.ndarray, pixel_threshold: float = 8.0, color_pixel_ratio: float = 0.02) -> bool:
    """
    픽셀별로 BGR 채널 간 표준편차를 구해서, 뚜렷하게 색이 있는 픽셀
    (표준편차 > pixel_threshold)의 비율이 color_pixel_ratio 미만이면 흑백으로 판단.

    이미지 전체 평균만 보면(예전 방식) 국소적으로 존재하는 진짜 색이 나머지 무채색
    영역에 희석돼 사라지는 문제가 있었다(실측: 금속에 반사된 색이 전체 픽셀의 4.7%를
    차지하는 사진도 평균은 낮아서 흑백으로 오판됨). 그렇다고 "색 픽셀이 조금이라도
    있으면 컬러"로 하면 JPEG 압축 노이즈만으로도 진짜 흑백 사진까지 컬러로 오판하게
    된다(실측: 그레이스케일→JPEG 재인코딩한 진짜 흑백 사진 30장은 색 픽셀 비율 0%,
    실제 컬러 사진 30장은 대부분 60~94%로 명확히 갈림) — 그래서 노이즈 수준은
    허용하되 일정 비율 이상 색 픽셀이 몰려있으면 컬러로 보는 절충안을 쓴다.
    """
    per_pixel_std = np.std(image.astype("float32"), axis=2)
    ratio = float(np.mean(per_pixel_std > pixel_threshold))
    return ratio < color_pixel_ratio


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
    is_bw = is_bw_image(image)

    return {
        "brightness": brightness_percentile,
        "brightness_label": brightness_label,
        "color_tone": color_tone,
        "saturation": saturation_percentile,
        "saturation_label": saturation_label,
        "is_bw": is_bw,
    }
