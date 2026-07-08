"""
TAG_LENS - 실내/실외 데이터셋에 Places365 다양성 추가

기존 dataset_indoor(HuggingFace IndoorOutdoorNet-20K + 도시 사진)는 일반적인
캐주얼 사진 위주라, 반사 유리 건물/화보 촬영처럼 스타일이 다른 사진에 약함.
Places365(MIT, 365개 장면 카테고리, 180만 장)를 스트리밍으로 받아
공식 IO(indoor/outdoor) 매핑에 따라 라벨링해서 스타일 다양성을 추가한다.
전체(21GB)를 받지 않고 필요한 만큼만 스트리밍으로 샘플링한다.

사용법:
    python scripts/expand/expand_indoor_with_places365.py
"""

from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "dataset_indoor"
IO_MAP_PATH = Path(__file__).resolve().parent / "IO_places365.txt"

TARGET_PER_CLASS = 1000  # 클래스당 목표 (train 800 + val 200)
VAL_RATIO = 0.2
RANDOM_SEED = 42


def load_io_map() -> dict[str, int]:
    io_map = {}
    with open(IO_MAP_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, io = line.rsplit(" ", 1)
            name = path.split("/", 2)[-1].replace("_", " ")
            io_map[name] = int(io)
    return io_map


def main() -> None:
    random.seed(RANDOM_SEED)
    print("=" * 50)
    print("실내/실외 데이터셋에 Places365 다양성 추가")
    print("=" * 50)

    io_map = load_io_map()
    print(f"IO 매핑: {len(io_map)}개 카테고리")

    ds = load_dataset("ljnlonoljpiljm/places365-256px", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=20000, seed=RANDOM_SEED)
    hf_names = ds.features["label"].names

    train_dirs = {1: OUTPUT_DIR / "train" / "실내", 2: OUTPUT_DIR / "train" / "실외"}
    val_dirs = {1: OUTPUT_DIR / "val" / "실내", 2: OUTPUT_DIR / "val" / "실외"}
    for d in list(train_dirs.values()) + list(val_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    counts = {1: 0, 2: 0}
    target_total = TARGET_PER_CLASS
    print(f"\n클래스당 목표: {target_total}장 (train {int(target_total*(1-VAL_RATIO))} + val {int(target_total*VAL_RATIO)})")

    for example in ds:
        if counts[1] >= target_total and counts[2] >= target_total:
            break

        name = hf_names[example["label"]]
        io = io_map.get(name)
        if io is None or counts[io] >= target_total:
            continue

        is_val = random.random() < VAL_RATIO
        target_dir = (val_dirs if is_val else train_dirs)[io]
        example["image"].convert("RGB").save(target_dir / f"places_{name.replace('/', '-')}_{counts[io]}.jpg")
        counts[io] += 1

        if sum(counts.values()) % 200 == 0:
            print(f"  진행: 실내 {counts[1]}/{target_total}, 실외 {counts[2]}/{target_total}")

    print(f"\n완료: 실내 {counts[1]}장, 실외 {counts[2]}장 추가")

    for split, dirs in [("train", train_dirs), ("val", val_dirs)]:
        for io, d in dirs.items():
            label = "실내" if io == 1 else "실외"
            print(f"  최종 {split}/{label}: {len(list(d.iterdir()))}장")


if __name__ == "__main__":
    main()
