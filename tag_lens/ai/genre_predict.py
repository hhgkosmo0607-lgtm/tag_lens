"""
장르 5-class 분류(#자연/#도시/#음식/#사물/#인물, 확신도 낮으면 #기타) — CLIP 제로샷.

2026-07-09: 자체 학습 CNN(MobileNetV2 fine-tuning, Caltech-101/자체 수집 데이터)을
CLIP로 교체함. 오늘 다른 CNN들(indoor/daynight/animal)을 CLIP으로 바꾸면서 검증된
것과 같은 실패 패턴이 genre CNN에도 있음을 확인:
- 눈/모래 위 타이어 자국 클로즈업 → CNN "자연" 98.6%(오답) / CLIP "도시" 65.0%(정답)
- TSINGTAO 매장 건물 파사드 → CNN "사물" 58%("도시"는 후보에도 안 들어감) / CLIP "도시" 80.8%
- ZARA 매장 간판 클로즈업 → CNN "자연" 94%(오답) / CLIP "사물" 61.3%
공통 원인: 하늘/사람/음식처럼 장르를 규정하는 핵심 피사체가 없는 "맥락 정보 부족한
클로즈업/텍스처" 사진에서, CNN이 좁은 학습 데이터의 색감/질감 상관관계에 기대 확신도
높게 틀림. CLIP은 훨씬 넓은 사전학습 데이터 덕에 이런 케이스에서 더 견고함.

단, CLIP도 만능은 아님 — 장노출/카메라 흔들림 추상 사진 하나는 CLIP도 89.8% 확신도로
"사물"이라고 오답함(사용자 확인 후 알려진 한계로 남기고 넘어가기로 함). 기존 CNN처럼
LOW_CONFIDENCE_THRESHOLD(0.6) 미만이면 "기타"로 보내는 안전장치는 그대로 유지 —
CLIP도 확신 없는 사진까지 5개 장르 중 하나로 억지로 우기지 않게 하기 위함.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["자연", "도시", "음식", "사물", "인물"]

# CLIP이 5개 장르 중 하나를 억지로 고르지 않고, 확신이 낮으면 "기타"로 분류한다.
# (예: 애매한 클로즈업/아웃포커스 사진이 51%짜리 확률로 엉뚱한 장르에 배정되는 문제)
LOW_CONFIDENCE_THRESHOLD = 0.6

_GENRE_PROMPTS = [
    "a photo of nature, such as a forest, mountain, or beach landscape",
    "a photo of a city or urban street scene",
    "a photo of food",
    "a photo of an object or product",
    "a photo of a person",
]


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


def predict_genre(image, model_path: str | None = None) -> tuple[str, dict[str, float]]:
    """장르를 CLIP 제로샷으로 판별한다. 확신도가 LOW_CONFIDENCE_THRESHOLD 미만이면 "기타".

    model_path는 기존 호출부(app.py, reclassify_photos.py)와의 인터페이스 호환을 위해
    남겨뒀지만 CLIP은 로컬 .h5 경로가 필요 없어 쓰지 않는다.
    """
    try:
        import torch

        from ai.clip_utils import get_clip, image_to_pil

        model, processor = get_clip()
        pil_image = image_to_pil(image)

        inputs = processor(text=_GENRE_PROMPTS, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            scores = model(**inputs).logits_per_image.softmax(dim=1)[0]

        probs = {name: float(scores[idx]) for idx, name in enumerate(CLASS_NAMES)}
        genre = CLASS_NAMES[int(scores.argmax())]
    except Exception as e:
        print(f"[ERROR] Failed to predict genre via CLIP: {type(e).__name__}: {e}")
        return _heuristic_predict(image)

    if probs[genre] < LOW_CONFIDENCE_THRESHOLD:
        genre = "기타"
    return genre, probs


def resolve_genre_with_animal(genre: str, animal: str | None, face_count: int) -> str:
    """
    동물 태그가 붙을 정도로 동물이 뚜렷하게 찍힌 사진은, 장르 CNN이 뭘 골랐든(어떤
    장르든) "기타"로 보정한다. "동물"을 별도 장르로 만들지 않는 이유는 CNN 확률
    (genre_probs)이 5-class 그대로라 근거 없는 장르를 새로 만들면 확률 표시와
    모순되기 때문 — #고양이 등 동물 태그가 이미 "무엇인지"를 알려주므로
    "#기타 #고양이" 조합으로 충분하다.

    2026-07-08: 처음엔 "인물" 오분류만 보정했음 — 장르 CNN의 "인물" 학습 데이터가
    전부 사람 얼굴 클로즈업(정면 응시, 큰 눈, 얼굴 중심 구도)이라, 같은 구도의 동물
    클로즈업(특히 고양이)을 사람으로 착각하는 사례가 확인됨(실측: 고양이 사진이
    96.6% 확신도로 인물 판정).
    2026-07-09: "동물→음식" 오분류(의자 위 고양이가 62% 확신도로 "음식" 판정)가 새로
    발견돼서, 특정 장르(인물)만이 아니라 동물 태그가 붙으면 장르 불문 전부 보정하도록
    범위를 넓힘.

    face_count == 0 조건은 유지: mediapipe가 진짜 사람 얼굴을 감지했다면(face_count > 0)
    genre는 이미 얼굴 감지 단계에서 "인물"로 확정된 것이라(app.py 참고), 동물이 같이
    있어도(사람이 반려동물과 함께 찍힌 사진 등) 그 "인물" 판정은 신뢰할 근거가 있으므로
    보정하지 않는다. 인물이 아닌 다른 장르(음식/자연/사물/도시)는 애초에 얼굴 감지
    경로를 거치지 않아 face_count가 항상 0이라 이 조건이 실질적으로 걸리지 않는다.
    """
    if genre != "기타" and face_count == 0 and animal is not None:
        return "기타"
    return genre
