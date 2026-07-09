"""
동물 세부 태그(#강아지/#고양이/#말/#동물) — CLIP 제로샷.

2026-07-09: 원래는 ImageNet-1000 사전학습 MobileNetV2의 top-1 클래스 인덱스를
강아지(151~268)/고양이(281~285)/말(339 sorrel) 범위로 매핑해 재활용했으나(순수
사전학습, 직접 학습 안 함 — embedding.py와 같은 부류), 두 가지 오탐이 실사용
중 발견됨: (1) 특이 자세의 고양이가 "Ibizan_hound"(개 품종) 76% 확신도로 오분류,
(2) 초록 잎 속 빨간 장미가 "macaw"(마카오앵무) 41%로 동물 취급됨. ImageNet은
동물 "종"이 아니라 "품종"을 세밀하게 구분하도록 학습돼서(개 품종 118개 vs 고양이
품종 5개), 이런 편향이 생기기 쉽다. CLIP 제로샷으로 교체.

CLIP 적용 과정에서 실측으로 확인된 세 가지 함정과 대응:
1. 부정문 프롬프트("동물이지만 개/고양이/말은 아님")를 "기타 동물" + "비동물 거름망"
   겸용으로 썼더니, CLIP이 부정문을 잘 못 다뤄서 장미 사진이 "동물"에 41.6%나 걸림
   → "기타 동물"을 새/야생동물처럼 구체적인 긍정문으로 분리하고, "비동물"도
   식물/사물처럼 구체적인 긍정문으로 표현해서 해결.
2. 그것만으론 부족함 — 커피잔 사진(배경에 작은 동물 그림이 있는 원두 카드)이 96%
   확신도로 "새"로 오판됨. 원인: 이 사진은 사실 어떤 프롬프트와도 강하게 안 맞는데
   (원본 유사도 로짓 17~21점대로 전부 낮음), 그나마 제일 덜 안 맞는 "새"가 다른 후보들과
   상대 비교(softmax)로 부풀려져 45%까지 뻥튀기됨 — 절대적으로 약한 신호도 상대
   비교에서는 "1등"이 될 수 있다는 것. 정탐(고양이/말/새) 사진들의 최고 유사도 로짓은
   전부 24~28점대였던 반면, 이런 오탐들은 21점을 못 넘었음 — 그 사이인 23점을
   절대 임계값으로 추가해서, 상대 비교(동물 vs 비동물)와 절대 임계값을 모두 통과해야만
   동물로 판정하도록 이중 게이트를 둠. 8/8 케이스로 검증 완료.
3. 2026-07-09: 그래도 못 거른 오탐 3건 발견 — 완전히 새까만 야경 반사 사진(실루엣 하나가
   "새"로 오판, 로짓 24.99로 정탐 구간 안에 들어감), 빛번짐 장노출 추상사진("새" 23.32),
   구름/해안 풍경 사진("새" 23.08) — 셋 다 절대 임계값(23점)을 통과할 만큼 로짓이 높아서
   (2)의 방어로도 못 잡음. 근본 원인: CLIP은 사진 전체와 프롬프트 문장의 전역 유사도만
   재기 때문에, 진짜 동물이 없어도 사진의 색감/구도가 "동물 사진들"의 통계적 패턴과
   막연히 겹치면 무언가는 "1등"으로 뽑힘 — 이건 절대 임계값을 아무리 올려도 못 없앰.
   대신 "1등과 2등 종(species) 프롬프트 로짓 차이(gap)"를 봤더니 훨씬 깔끔하게 갈림:
   진짜 동물 사진은 정답 하나만 확 튀고 나머지 종은 한참 낮은 반면(정탐 8건 gap 3.68~5.68),
   위 오탐 3건은 개/고양이/말/새가 다 고만고만하게 붙어있었음(gap 0.80~1.56) — 실제
   동물이 있으면 CLIP도 "이건 새고 개/고양이/말은 확실히 아님"으로 명확히 갈리는데,
   오탐은 여러 종 프롬프트에 막연히 다 조금씩 걸치기 때문. gap 2.5를 임계값으로 추가.

4. 2026-07-09 추가: 위 세 방어를 다 통과한 "정탐"인데도, 동물이 화면 대비 점 하나
   수준으로만 찍힌 경우(노을진 바다 사진 속 새 한 마리, 화면의 0.02% 미만)까지 태그가
   붙는 게 과하다는 피드백 — CLIP은 전역 유사도만 재서 크기 정보가 아예 없어 이런
   "곁다리 동물"을 못 거름(사진을 축소해서 확신도가 떨어지는지 시도했으나 실패 — 축소
   강건성이 크기와 무관했고, 오히려 확실한 승마 사진이 32px 축소에서 확신도 폭락함).
   대신 mediapipe Object Detector(EfficientDet-Lite0, COCO 90-class 사전학습, 이미
   쓰던 mediapipe 라이브러리의 다른 Task)로 실제 바운딩박스를 얻어 크기를 직접 잼 —
   낮은 임계값(0.02)으로 최대 재현율로 감지한 뒤, COCO 동물 카테고리 중 최대 박스
   면적 비율을 계산. 실측: 점만 한 새는 0.047%(최선의 감지조차 이 정도), 확실한
   동물 사진들(고양이 6장 4.4~71.2%, 말 1장은 이 모델이 정확히 "말"로는 잘 못 잡지만
   그 자리의 박스 면적은 7.7%)은 전부 1% 이상 — 격차가 165배라 1%를 임계값으로 설정.
   predict_animal()이 이미 종을 확정한 뒤 마지막 관문으로 추가, 감지기 자체가 실패하면
   (모델 로드 불가 등) 안전하게 통과시킴(태그를 함부로 없애지 않음).
"""

from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.3

_SPECIES_PROMPTS = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a bird",
    "a photo of a wild animal such as a lion, elephant, or deer",
]
_SPECIES_LABELS = ["강아지", "고양이", "말", "동물", "동물"]

_NON_ANIMAL_PROMPTS = [
    "a photo of a plant, flower, or tree",
    "a photo of a person, building, or object with no animal",
]

# 세 조건을 모두 만족해야 동물로 판정한다(실측으로 확인된 CLIP의 세 가지 함정을 각각 방어):
# (1) 상대 비교: 동물 프롬프트 합산 확률 > 비동물 프롬프트 합산 확률 — softmax 기반,
#     "동물 vs 비동물" 중 뭐가 더 그럴듯한지 비교.
# (2) 절대 임계값: 최고 매칭 프롬프트의 원본 유사도 로짓(softmax 전)이 이 값 이상 —
#     (1)만으로는 "둘 다 약하게 안 맞지만 그나마 동물 쪽이 상대적으로 나음"인 경우까지
#     동물로 오판하게 됨(실측: 정탐 사진 최고 로짓 24~28점대, 오탐은 21점 이하).
_ANIMAL_LOGIT_THRESHOLD = 23.0

# (3) 종(species) 프롬프트 1등-2등 로짓 차이 — (1)(2)를 다 통과하고도 남는 오탐 방어.
#     진짜 동물 사진은 정답 종 하나만 확 튀고 나머지는 한참 낮은 반면(정탐 실측
#     gap 3.68~5.68), 동물이 없는데 전체적인 색감/구도가 막연히 "동물틱"해서 걸리는
#     사진은 개/고양이/말/새가 다 고만고만하게 붙어있음(오탐 실측 gap 0.80~1.56).
_ANIMAL_SPECIES_GAP_THRESHOLD = 2.5

# (4) 화면 대비 크기 — CLIP이 종까지 확정해도, 동물이 점 하나 수준으로 작으면
#     (사진의 주제가 아니라 곁다리) 태그를 안 붙인다. mediapipe Object Detector로
#     실제 바운딩박스 면적을 재서 판단(실측: 점만 한 새 0.047% vs 확실한 동물들 4.4%
#     이상 — 165배 격차, 1%를 그 사이 임계값으로 설정).
_ANIMAL_COCO_CATEGORIES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}
_MIN_ANIMAL_AREA_RATIO = 0.01
_OBJECT_DETECT_SCORE_THRESHOLD = 0.02  # 크기만 잴 거라 재현율 우선(정확한 종 분류는 CLIP이 이미 함)

_OBJECT_DETECTOR = None


def _get_object_detector():
    global _OBJECT_DETECTOR
    if _OBJECT_DETECTOR is None:
        from pathlib import Path

        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path(__file__).resolve().parent.parent / "models" / "efficientdet_lite0.tflite"
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.ObjectDetectorOptions(
            base_options=base_options, score_threshold=_OBJECT_DETECT_SCORE_THRESHOLD, max_results=20
        )
        _OBJECT_DETECTOR = mp_vision.ObjectDetector.create_from_options(options)
    return _OBJECT_DETECTOR


def _passes_size_gate(image) -> bool:
    """동물로 보이는 COCO 카테고리 박스 중 최대 면적 비율이 임계값 이상인지 확인한다.
    감지기 자체가 실패하면(모델 로드 불가 등) 태그를 함부로 없애지 않도록 통과시킨다."""
    try:
        import cv2
        import mediapipe as mp

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _get_object_detector().detect(mp_image)

        best_ratio = 0.0
        for detection in result.detections:
            if detection.categories[0].category_name in _ANIMAL_COCO_CATEGORIES:
                box = detection.bounding_box
                best_ratio = max(best_ratio, (box.width * box.height) / (w * h))
        return best_ratio >= _MIN_ANIMAL_AREA_RATIO
    except Exception as e:
        print(f"[ERROR] Failed to run animal size gate: {type(e).__name__}: {e}")
        return True


def predict_animal(image) -> str | None:
    """
    이미지가 동물 클래스에 속하는지 CLIP 제로샷으로 판단해 태그를 반환한다.
    동물이 아니거나 확신도가 낮으면 None.
    반환값: "강아지" | "고양이" | "말" | "동물" | None
    """
    try:
        import torch

        from ai.clip_utils import get_clip, image_to_pil

        model, processor = get_clip()
        pil_image = image_to_pil(image)

        prompts = _SPECIES_PROMPTS + _NON_ANIMAL_PROMPTS
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
            scores = logits.softmax(dim=0)

        species_logits = logits[: len(_SPECIES_PROMPTS)]
        species_scores = scores[: len(_SPECIES_PROMPTS)]
        nonanimal_scores = scores[len(_SPECIES_PROMPTS) :]

        best_idx = int(species_logits.argmax())
        best_logit = float(species_logits[best_idx])
        animal_presence = float(species_scores.sum())
        nonanimal_presence = float(nonanimal_scores.sum())
        second_logit = float(sorted(species_logits.tolist())[-2])
        species_gap = best_logit - second_logit

        if (
            best_logit < _ANIMAL_LOGIT_THRESHOLD
            or animal_presence <= nonanimal_presence
            or species_gap < _ANIMAL_SPECIES_GAP_THRESHOLD
        ):
            return None

        best_conf = float(species_scores[best_idx])
        if best_conf < CONFIDENCE_THRESHOLD:
            return None
        if not _passes_size_gate(image):
            return None
        return _SPECIES_LABELS[best_idx]
    except Exception as e:
        print(f"[ERROR] Failed to predict animal: {type(e).__name__}: {e}")
        return None
