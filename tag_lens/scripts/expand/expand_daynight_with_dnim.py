"""
TAG_LENS - 주간/야간 데이터셋에 DNIM(다지점 웹캠) 다양성 추가

기존 dataset_daynight(Kaggle ibrahimalobaid, 도로 사진 1곳)는 도메인이 좁아서
일반화가 안 됨. Kaggle stevemark/daynight-dataset(DNIM, Archive of Many Outdoor
Scenes 기반)은 17개의 서로 다른 웹캠 위치에서 찍힌 사진이라 훨씬 다양하다.

밝기 기반 pseudo-label(이전 시도, 실패)과 달리, 파일명에 박힌 실제 촬영 시각
(예: 20151101_072507.jpg = 2015-11-01 07:25:07)으로 라벨링한다 — 이건 추정이
아니라 실측값이라 신뢰도가 훨씬 높다. 새벽/황혼처럼 애매한 시간대는 제외하고
확실히 낮/밤인 시간대만 사용한다.

사용법:
    python scripts/expand/expand_daynight_with_dnim.py
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

import kagglehub

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "dataset_daynight"

DAY_HOURS = set(range(8, 17))       # 08:00 ~ 16:59 확실한 낮
NIGHT_HOURS = {21, 22, 23, 0, 1, 2, 3, 4}  # 21:00 ~ 04:59 확실한 밤
# 05~07시, 17~20시(여명/황혼)는 애매해서 제외

VAL_RATIO = 0.2
RANDOM_SEED = 42

FILENAME_RE = re.compile(r"_(\d{2})\d{4}\.jpg$")


def main() -> None:
    random.seed(RANDOM_SEED)
    print("=" * 50)
    print("주간/야간 데이터셋에 DNIM 다양성 추가")
    print("=" * 50)

    print("\n[1/2] DNIM 데이터셋 다운로드 중...")
    dnim_root = Path(kagglehub.dataset_download("stevemark/daynight-dataset"))
    image_dir = dnim_root / "DNIM" / "Image"
    print(f"  -> {image_dir}")

    day_files: list[Path] = []
    night_files: list[Path] = []
    for f in image_dir.glob("*/*.jpg"):
        m = FILENAME_RE.search(f.name)
        if not m:
            continue
        hour = int(m.group(1))
        if hour in DAY_HOURS:
            day_files.append(f)
        elif hour in NIGHT_HOURS:
            night_files.append(f)

    print(f"  주간(08-16시): {len(day_files)}장, 야간(21-04시): {len(night_files)}장")

    print("\n[2/2] dataset_daynight/에 추가 중...")
    for label, files in [("주간", day_files), ("야간", night_files)]:
        random.shuffle(files)
        val_count = int(len(files) * VAL_RATIO)
        val_files = files[:val_count]
        train_files = files[val_count:]

        train_dir = OUTPUT_DIR / "train" / label
        val_dir = OUTPUT_DIR / "val" / label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(train_files):
            shutil.copy(f, train_dir / f"dnim_{i}.jpg")
        for i, f in enumerate(val_files):
            shutil.copy(f, val_dir / f"dnim_{i}.jpg")

        print(f"  {label}: train {len(train_files)}장, val {len(val_files)}장 추가")

    for split in ("train", "val"):
        for label in ("주간", "야간"):
            d = OUTPUT_DIR / split / label
            print(f"  최종 {split}/{label}: {len(list(d.iterdir()))}장")

    print("\n완료!")


if __name__ == "__main__":
    main()
