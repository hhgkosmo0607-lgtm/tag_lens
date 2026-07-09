from __future__ import annotations

import numpy as np

CLASS_NAMES = ["실내", "실외"]

# genre_predict.py의 LOW_CONFIDENCE_THRESHOLD와 같은 이유 — 51% vs 49%처럼 애매한
# 경우까지 억지로 실내/실외를 확정하지 않고, 확신이 낮으면 판별 자체를 건너뛴다.
LOW_CONFIDENCE_THRESHOLD = 0.6

# 2026-07-08~09: 실내/실외 CNN이 자연 장르에서 오탐이 잦아(36장 중 6장) 도시/인물
# 장르에만 적용하는 화이트리스트(INDOOR_CHECK_GENRES = {"도시","인물"})를 뒀었다.
# 2026-07-09 CLIP 전환 후 재검증: 사물/음식/기타 장르 샘플(카페 벽 콜라주, 뱅크시
# 벽화, 모나리자 포스터 매대 등)에서도 합리적인 결과가 나왔고, 자연 장르 9장 전부
# 87.7~100% 확신도로 정확히 실외 판정됨(오탐 0건) — CNN 시절의 자연 오탐 패턴이
# CLIP에는 재현되지 않아, 장르 화이트리스트를 없애고 모든 장르에 적용하도록 변경.
# 확신도가 낮으면(LOW_CONFIDENCE_THRESHOLD 미만) 어차피 태그가 안 붙으므로, 사물/
# 음식처럼 공간 맥락이 부족한 클로즈업 사진은 그 경우 자연스럽게 걸러진다.

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

# 2026-07-09 프롬프트 개정: "밤에 조명만 밝고 하늘이 안 보이는 실외"(가게 앞에 세워둔
# 자전거, 환하게 불 켜진 매장 전면이 배경을 채움) 사진이 실내 89.2%로 오판되는 사례
# 발견 — 기존 프롬프트("실내 장면, 천장과 벽이 있는")가 "카메라가 실내에 있는지"가
# 아니라 "장면에 실내스러운 디테일(조명/진열)이 보이는지"에 반응하는 것으로 추정.
# "카메라/촬영자의 위치"를 명시하는 프롬프트로 교체해서 해결: 타겟 사진 실내 89.2%→
# 58.2%(임계값 미만으로 태그 제거), 기존 정탐 8건(실내 4 + 실외 4) + 자연 장르 9건
# 전부 회귀 없음 확인.
_INDOOR_PROMPTS = [
    "a photo taken from inside a room or building, surrounded by walls and a ceiling",
    "the camera is indoors, inside an enclosed space",
]
_OUTDOOR_PROMPTS = [
    "a photo taken while standing outdoors on a street, sidewalk, or open area",
    "the camera is outdoors in the open air, not enclosed by walls",
]

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
