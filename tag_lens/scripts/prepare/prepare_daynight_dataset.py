"""
TAG_LENS - 주간/야간 이진 분류 데이터 준비 스크립트

Kaggle의 day-and-night-image 데이터셋(day_night_images/{training,test}/{day,night})을
주간/야간 2-class로 정리해 dataset_daynight/{train,val} 폴더를 만든다.

사전 준비:
    pip install kagglehub
    C:/Users/<사용자명>/.kaggle/kaggle.json 에 Kaggle API 토큰 저장

사용법:
    python prepare_daynight_dataset.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub

OUTPUT_DIR = Path("./dataset_daynight")
VAL_RATIO = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def download_dataset() -> Path:
    print("[1/3] Day/Night 이미지 다운로드 중...")
    path = kagglehub.dataset_download("ibrahimalobaid/day-and-night-image")
    print(f"  -> {path}")
    return Path(path) / "day_night_images"


def collect_images(root: Path) -> dict[str, list[Path]]:
    print("[2/3] 주간/야간 이미지 수집 중...")
    mapping = {"주간": "day", "야간": "night"}
    collected: dict[str, list[Path]] = {}
    for label, folder in mapping.items():
        images: list[Path] = []
        for split in ("training", "test"):
            src_dir = root / split / folder
            if src_dir.exists():
                images.extend(src_dir.glob("*.*"))
        random.shuffle(images)
        collected[label] = images
        print(f"  -> {label}: {len(images)}장")
    return collected


def split_train_val(collected: dict[str, list[Path]]) -> None:
    print("[3/3] train/val 분리 중...")
    for label, images in collected.items():
        n_val = int(len(images) * VAL_RATIO)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        train_dir = OUTPUT_DIR / "train" / label
        val_dir = OUTPUT_DIR / "val" / label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(train_imgs):
            shutil.copy(img, train_dir / f"{i}_{img.name}")
        for i, img in enumerate(val_imgs):
            shutil.copy(img, val_dir / f"{i}_{img.name}")

        print(f"  -> {label}: train {len(train_imgs)}장 / val {len(val_imgs)}장")


def main() -> None:
    root = download_dataset()
    collected = collect_images(root)
    split_train_val(collected)
    print("\n완료! 데이터셋 위치:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
