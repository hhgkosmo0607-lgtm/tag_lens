"""
CLIP 모델 공유 로더 — indoor_predict.py, daynight_predict.py, animal_predict.py가
전부 같은 CLIP 모델(openai/clip-vit-base-patch32)을 순수 사전학습 상태로 쓴다.

2026-07-09: 처음엔 각 모듈이 자체 전역 캐시(_CLIP_MODEL)로 따로 로드했는데, 세
모듈을 한 프로세스에서 같이 쓰는 reclassify_photos.py/app.py 입장에서는 동일한
모델을 메모리에 3번 중복 로드하는 낭비였음 — 하나로 합침.
"""

from __future__ import annotations

_MODEL_NAME = "openai/clip-vit-base-patch32"

_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip():
    """CLIP 모델/프로세서(전역 캐시). 최초 1회만 로드하고 이후 모든 호출부가 재사용한다."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor

        _CLIP_MODEL = CLIPModel.from_pretrained(_MODEL_NAME)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def image_to_pil(image):
    """OpenCV BGR numpy 배열을 CLIP 프로세서가 받는 PIL RGB 이미지로 변환한다."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
