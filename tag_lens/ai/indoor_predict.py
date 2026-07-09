from __future__ import annotations

import numpy as np

CLASS_NAMES = ["실내", "실외"]

# genre_predict.py의 LOW_CONFIDENCE_THRESHOLD와 같은 이유 — 51% vs 49%처럼 애매한
# 경우까지 억지로 실내/실외를 확정하지 않고, 확신이 낮으면 판별 자체를 건너뛴다.
LOW_CONFIDENCE_THRESHOLD = 0.6

# 실내/실외 판별은 배경이 어느 정도 보이는 장르에만 적용한다.
# 사물/음식은 클로즈업 정물 위주라 판단 근거(벽/천장 등 공간 맥락)가 사진에 거의 없어 제외.
#
# 자연도 2026-07-09에 제외로 변경: 실측(실제 업로드 사진 36장 중 자연 장르 6장에 #실내
# 오탐)으로 확인해보니 (1) 자연 장르 사진은 정의상 거의 항상 실외(숲/바다/산)라 진짜
# 실내인 경우(창문 너머 촬영 등)가 드문 반면, (2) 장노출/흔들림 추상 사진이 장르
# CNN에서 자연으로 오분류된 경우와 (3) 야간 실외 사진(불빛만 밝고 하늘이 안 보임)이
# 실내 CNN에 반복적으로 실내로 오판되는 경우가 훨씬 흔했음(주간 33%, 야간 56%가
# #실내 오탐). 도시/인물은 스튜디오·카페처럼 실내/실외 구분 자체가 의미 있는 장르라 유지.
INDOOR_CHECK_GENRES = {"도시", "인물"}

# 2026-07-09: 자체 학습 CNN(MobileNetV2 fine-tuning)을 CLIP 제로샷으로 교체함.
#
# 경위: "하늘 안 보이는 실외"(건물 벽/간판 클로즈업, 반사 유리, 다리, 야시장) 데이터를
# 보강해서 재학습을 시도했으나(scripts/expand/expand_indoor_no_sky.py), 검증셋 정확도는
# 97.49%로 좋아 보였음에도 실제 사진 15장 재검증에서 13.3%(거의 전부 "실내"로 오판,
# 확신도 90~100%)로 오히려 더 나빠짐 — 보강 데이터(웹캠 야간 사진, 크롭된 근접 텍스처)의
# 좁고 특정한 시각적 패턴에 과적합된 것으로 추정. 재학습을 롤백하고, 같은 사진 15장으로
# CLIP 제로샷("a photo taken indoors" vs "a photo taken outdoors" 유사 프롬프트)을
# 테스트했더니 정확도 73.3%(오탐률 100%→28.6%)로 훨씬 안정적임을 확인해 교체함.
#
# 다만 이 15장은 이미 알려진 실패 패턴(하늘 안 보이는 실외)에 맞춰 고른 사진들이라,
# "기존 CNN이 전반적으로 100% 오탐"이라는 뜻은 아님 — 그 특정 패턴에서만 CNN이
# 확신도 높게 틀렸고, CLIP은 같은 패턴에서 상대적으로 더 견고했다는 의미로 해석할 것.
#
# ai/embedding.py, ai/animal_predict.py와 같은 "순수 사전학습, 직접 학습 안 함" 계열 —
# CLIP도 이 프로젝트에서 파인튜닝하지 않고 그대로 쓴다.
_MODEL_NAME = "openai/clip-vit-base-patch32"
_INDOOR_PROMPTS = ["a photo taken indoors, inside a building", "an indoor scene with a ceiling and walls"]
_OUTDOOR_PROMPTS = ["a photo taken outdoors, outside a building", "an outdoor scene in the open air"]

_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _get_clip():
    """CLIP 모델/프로세서(전역 캐시). 최초 1회만 로드하고 이후 재사용한다."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor

        _CLIP_MODEL = CLIPModel.from_pretrained(_MODEL_NAME)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _heuristic_predict(image) -> tuple[str, dict[str, float]]:
    """CLIP을 못 쓸 때의 약한 대체 판별: 사진 상단 1/3에 하늘색(밝고 파란) 영역이 많으면 실외로 추정."""
    h, w = image.shape[:2]
    top_third = image[: h // 3, :, :]
    b, g, r = np.mean(top_third.reshape(-1, 3), axis=0)
    sky_score = max(0.0, min(1.0, ((b - r) / 255.0) + (float(np.mean(top_third)) / 255.0) - 0.5))
    probs = {"실외": sky_score, "실내": 1.0 - sky_score}
    label = max(probs, key=probs.get)
    return label, probs


def predict_indoor(image, model_path: str | None = None) -> tuple[str | None, dict[str, float]]:
    """실내/실외를 판별한다. 확신도가 LOW_CONFIDENCE_THRESHOLD 미만이면 판별을 건너뛰고
    (None, probs)를 반환한다 — tagging.py는 indoor가 None이면 태그를 붙이지 않는다.

    model_path는 기존 호출부(app.py, reclassify_photos.py)와의 인터페이스 호환을 위해
    남겨뒀지만 CLIP은 로컬 .h5 경로가 필요 없어 쓰지 않는다.
    """
    try:
        import cv2
        import torch
        from PIL import Image

        model, processor = _get_clip()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prompts = _INDOOR_PROMPTS + _OUTDOOR_PROMPTS
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            scores = model(**inputs).logits_per_image.softmax(dim=1)[0]

        indoor_score = float(scores[: len(_INDOOR_PROMPTS)].sum())
        outdoor_score = float(scores[len(_INDOOR_PROMPTS) :].sum())
        probs = {"실내": indoor_score, "실외": outdoor_score}
        label = max(probs, key=probs.get)
    except Exception as e:
        print(f"[ERROR] Failed to predict indoor/outdoor via CLIP: {type(e).__name__}: {e}")
        label, probs = _heuristic_predict(image)

    if probs[label] < LOW_CONFIDENCE_THRESHOLD:
        return None, probs
    return label, probs
