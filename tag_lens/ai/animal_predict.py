"""
동물 세부 태그(#강아지/#고양이/#말/#동물) — 순수 ImageNet 사전학습 MobileNetV2를 그대로 쓴다.
embedding.py와 같은 부류(우리가 학습/fine-tuning한 모델이 아님)지만, 여기서는
include_top=True로 1000-class 분류 결과를 그대로 읽는다.

ImageNet-1k는 클래스 인덱스 0~397번이 전부 동물(어류/조류/파충류/곤충/포유류 등)이고
398번(abacus)부터 사물/장면 클래스로 넘어간다 — imagenet_class_index.json을 직접
훑어서 확인한 경계값이다. 그 안에서 강아지(151~268)와 고양이(281~285)는 넓은 범위를
차지하고, 말은 "horse"라는 이름의 클래스가 따로 없고 밤색 말을 뜻하는 "sorrel"(339)이
사실상 말 클래스다(얼룩말은 340 zebra로 별개). 이 세 가지만 세분화하고 나머지
동물 클래스(0~397, 위 세 범위 제외)는 전부 "#동물"로 뭉뚱그린다.
"""

from __future__ import annotations

import numpy as np

DOG_INDICES = range(151, 269)
CAT_INDICES = range(281, 286)
HORSE_INDEX = 339
ANIMAL_MAX_INDEX = 397  # 0~397번이 ImageNet의 동물 클래스 전체 (398 abacus부터 사물/장면)

CONFIDENCE_THRESHOLD = 0.3

_ANIMAL_MODEL = None


def _get_animal_model():
    """동물 분류 모델(전역 캐시). 최초 1회만 로드하고 이후 재사용한다."""
    global _ANIMAL_MODEL
    if _ANIMAL_MODEL is None:
        from tensorflow.keras.applications import MobileNetV2

        _ANIMAL_MODEL = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    return _ANIMAL_MODEL


def predict_animal(image) -> str | None:
    """
    이미지의 ImageNet top-1 예측이 동물 클래스인지 판단해 태그를 반환한다.
    동물이 아니거나 확신도가 낮으면 None.
    반환값: "강아지" | "고양이" | "말" | "동물" | None
    """
    try:
        import cv2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        model = _get_animal_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224)).astype("float32")
        batch = preprocess_input(np.expand_dims(resized, axis=0))
        preds = model.predict(batch, verbose=0)[0]

        top_index = int(np.argmax(preds))
        top_confidence = float(preds[top_index])

        if top_confidence < CONFIDENCE_THRESHOLD or top_index > ANIMAL_MAX_INDEX:
            return None

        if top_index in DOG_INDICES:
            return "강아지"
        if top_index in CAT_INDICES:
            return "고양이"
        if top_index == HORSE_INDEX:
            return "말"
        return "동물"
    except Exception as e:
        print(f"[ERROR] Failed to predict animal: {type(e).__name__}: {e}")
        return None
