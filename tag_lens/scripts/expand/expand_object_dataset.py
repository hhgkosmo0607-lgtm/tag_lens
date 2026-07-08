"""
TAG_LENS - 사물 데이터셋에 카테고리 다양성 추가

기존 dataset/train,val/사물은 Caltech-101의 16개 카테고리(대부분 흰 배경 정물)뿐이라
실사용 사물 사진 대부분을 커버 못 함. 이미 캐시된 Caltech-101에서 새 카테고리
18개(악기/생활용품/스포츠용품 등)를 추가로 뽑아 기존 dataset/train,val/사물에 합친다.
prepare_genre_dataset.py의 OBJECT_CATEGORIES 확장분과 동일한 카테고리를 사용한다.

사용법:
    python scripts/expand/expand_object_dataset.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
VAL_RATIO = 0.2
RANDOM_SEED = 42

NEW_CATEGORIES = [
    "accordion", "anchor", "barrel", "bass", "ceiling_fan", "chandelier",
    "dollar_bill", "electric_guitar", "euphonium", "gramophone",
    "grand_piano", "inline_skate", "menorah", "metronome", "saxophone",
    "soccer_ball", "stop_sign", "wheelchair",
]


def find_class_dir(root: Path, class_name: str) -> Path | None:
    matches = [p for p in root.rglob(class_name) if p.is_dir()]
    return matches[0] if matches else None


def main() -> None:
    random.seed(RANDOM_SEED)
    print("=" * 50)
    print("사물 데이터셋 확장 (Caltech-101 카테고리 추가)")
    print("=" * 50)

    caltech_root = Path(kagglehub.dataset_download("imbikramsaha/caltech-101"))
    print(f"Caltech-101: {caltech_root}\n")

    train_dir = DATASET_DIR / "train" / "사물"
    val_dir = DATASET_DIR / "val" / "사물"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_train, total_val = 0, 0
    for cat in NEW_CATEGORIES:
        src_dir = find_class_dir(caltech_root, cat)
        if src_dir is None:
            print(f"  ! 경고: '{cat}' 카테고리를 찾지 못함 (건너뜀)")
            continue

        files = list(src_dir.glob("*.jpg"))
        random.shuffle(files)
        val_count = int(len(files) * VAL_RATIO)
        val_files = files[:val_count]
        train_files = files[val_count:]

        for img in train_files:
            shutil.copy(img, train_dir / f"{cat}_{img.name}")
        for img in val_files:
            shutil.copy(img, val_dir / f"{cat}_{img.name}")

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"  {cat}: train {len(train_files)}장, val {len(val_files)}장")

    print(f"\n총 추가: train {total_train}장, val {total_val}장")
    print(f"최종 사물: train {len(list(train_dir.iterdir()))}장, val {len(list(val_dir.iterdir()))}장")
    print("\n완료!")


if __name__ == "__main__":
    main()
