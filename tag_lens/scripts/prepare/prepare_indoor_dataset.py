"""
TAG_LENS - 실내/실외 이진 분류 데이터 준비 스크립트

HuggingFace의 prithivMLmods/IndoorOutdoorNet-20K(약 2만 장, Apache-2.0)를
실내/실외 2-class로 정리해 dataset_indoor/{train,val} 폴더를 만든다.
day/night 데이터셋과 달리 좁은 도메인(도로 CCTV 등)이 아니라 일반적인
다양한 장면 사진이라, 자연/도시/인물처럼 배경이 보이는 장르에 적용하기 적합하다.

사전 준비:
    pip install datasets

사용법:
    python scripts/prepare/prepare_indoor_dataset.py
"""

from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "dataset_indoor"
VAL_RATIO = 0.2
RANDOM_SEED = 42

LABEL_MAP = {"Indoor": "실내", "Outdoor": "실외"}


def main() -> None:
    random.seed(RANDOM_SEED)
    print("=" * 50)
    print("TAG_LENS - 실내/실외 데이터 준비")
    print("=" * 50)

    print("\n[1/2] HuggingFace에서 IndoorOutdoorNet-20K 다운로드 중...")
    ds = load_dataset("prithivMLmods/IndoorOutdoorNet-20K", split="train")
    label_names = ds.features["label"].names
    print(f"  -> {len(ds)}장, 라벨: {label_names}")

    print("\n[2/2] train/val 분리 및 저장 중...")
    indices_by_label: dict[str, list[int]] = {name: [] for name in label_names}
    for idx, label_id in enumerate(ds["label"]):
        indices_by_label[label_names[label_id]].append(idx)

    for en_label, indices in indices_by_label.items():
        ko_label = LABEL_MAP[en_label]
        random.shuffle(indices)
        val_count = int(len(indices) * VAL_RATIO)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

        train_dir = OUTPUT_DIR / "train" / ko_label
        val_dir = OUTPUT_DIR / "val" / ko_label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(train_indices):
            ds[idx]["image"].convert("RGB").save(train_dir / f"{en_label.lower()}_{i}.jpg")
        for i, idx in enumerate(val_indices):
            ds[idx]["image"].convert("RGB").save(val_dir / f"{en_label.lower()}_{i}.jpg")

        print(f"  {ko_label}: train {len(train_indices)}장, val {len(val_indices)}장")

    print("\n완료! 데이터셋 위치:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
