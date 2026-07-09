"""
TAG_LENS - 실내/실외 데이터셋에 "하늘 안 보이는 실외" 사례 보강

실사용 테스트(2026-07-09)에서 실내/실외 CNN이 건물 벽/간판 클로즈업, 반사 유리 건물,
다리, 야시장처럼 "인공 구조물이 화면을 채우고 하늘이 안 보이는" 실외 사진을 90%+
확신도로 실내로 오판하는 게 광범위하게 확인됨(도시/인물 장르 실내 태그 14건 중 9건).
기존 학습 데이터(HuggingFace IndoorOutdoorNet-20K + Places365 + Intel 도시)가
하늘/풍경이 보이는 "일반적인" 실외 사진 위주라, 이런 구도 자체가 부족했던 것으로 보임.

두 가지 방법으로 "하늘 안 보이는 실외" 사례를 보강한다(새 데이터셋 다운로드 없이
기존 로컬 자산만 재활용):

1. dataset_daynight/야간(DNIM, 실외 웹캠 야간 촬영, 실측 확인됨)을 실외로 재활용
   — "야간 = 실내일 가능성 높다"는 부정확한 상관관계를 깨는 데 직접적으로 도움
2. 기존 dataset_indoor/실외 사진 일부를 상단(하늘이 있을 위치) 30~50%를 무작위
   비율로 잘라내 "하늘이 안 보이는 실외" 변형을 합성 — 모델이 "하늘 보임=실외"라는
   얕은 지름길 대신 다른 단서(원근감, 재질, 인공조명 패턴 등)를 보도록 강제

사용법:
    python scripts/expand/expand_indoor_no_sky.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INDOOR_DIR = BASE_DIR / "dataset_indoor"
DAYNIGHT_DIR = BASE_DIR / "dataset_daynight"

CROP_TRAIN_COUNT = 800
CROP_VAL_COUNT = 200
CROP_MIN_RATIO = 0.30  # 상단에서 잘라낼 최소 비율
CROP_MAX_RATIO = 0.50  # 상단에서 잘라낼 최대 비율
RANDOM_SEED = 42


def copy_daynight_night_images() -> None:
    """dataset_daynight의 야간(실외 웹캠) 이미지를 dataset_indoor/실외로 복사한다."""
    print("[1/2] dataset_daynight/야간 → dataset_indoor/실외 재활용")
    for split in ("train", "val"):
        src_dir = DAYNIGHT_DIR / split / "야간"
        dst_dir = INDOOR_DIR / split / "실외"
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in src_dir.glob("*"):
            if not src.is_file():
                continue
            dst = dst_dir / f"daynight_night_{src.name}"
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            copied += 1
        print(f"  {split}: {copied}장 복사")


def crop_no_sky_variants() -> None:
    """기존 실외 사진 일부의 상단을 잘라내 '하늘 안 보이는 실외' 변형을 만든다."""
    print("\n[2/2] 실외 사진 상단 크롭으로 '하늘 안 보이는 실외' 변형 생성")
    random.seed(RANDOM_SEED)

    for split, target_count in (("train", CROP_TRAIN_COUNT), ("val", CROP_VAL_COUNT)):
        outdoor_dir = INDOOR_DIR / split / "실외"
        # 이미 크롭한 변형(_nosky)은 소스 후보에서 제외 — 크롭한 걸 또 크롭하지 않도록
        candidates = [p for p in outdoor_dir.glob("*") if p.is_file() and "_nosky" not in p.stem]
        sample = random.sample(candidates, min(target_count, len(candidates)))

        created = 0
        for src in sample:
            try:
                image = Image.open(src).convert("RGB")
            except Exception:
                continue
            w, h = image.size
            crop_ratio = random.uniform(CROP_MIN_RATIO, CROP_MAX_RATIO)
            top = int(h * crop_ratio)
            if h - top < 32:  # 너무 작아지면 스킵
                continue
            cropped = image.crop((0, top, w, h))
            dst = outdoor_dir / f"{src.stem}_nosky{src.suffix}"
            cropped.save(dst)
            created += 1

        print(f"  {split}: {created}장 생성 (원본 {len(candidates)}장 중 샘플링)")


def main() -> None:
    print("=" * 50)
    print("실내/실외 데이터셋 - 하늘 안 보이는 실외 사례 보강")
    print("=" * 50)

    copy_daynight_night_images()
    crop_no_sky_variants()

    print("\n완료. 최종 실외 개수:")
    for split in ("train", "val"):
        d = INDOOR_DIR / split / "실외"
        print(f"  {split}/실외: {len(list(d.iterdir()))}장")
        d2 = INDOOR_DIR / split / "실내"
        print(f"  {split}/실내: {len(list(d2.iterdir()))}장")


if __name__ == "__main__":
    main()
