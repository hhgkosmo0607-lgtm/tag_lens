"""
TAG_LENS - CNN 학습 데이터 자동 준비 스크립트

이 스크립트는 다음을 자동으로 수행합니다:
1. Kaggle에서 Intel Image Classification, Food-101, Caltech-101 데이터셋 다운로드
2. 풍경(sea+glacier+mountain+forest) / 도시(buildings+street) / 음식(선택 카테고리)
   / 사물(선택 카테고리) 로 재분류 (4-class)
3. 클래스별 이미지 수를 맞추는 언더샘플링
4. train/val 8:2 분리

사전 준비:
    pip install kagglehub
    C:/Users/<사용자명>/.kaggle/kaggle.json 에 Kaggle API 토큰 저장
    (https://www.kaggle.com/settings/api 에서 발급 후 저장)

사용법:
    python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

import kagglehub

# ---------- 설정 ----------
OUTPUT_DIR = Path("./dataset")          # 최종 결과가 저장될 경로
SAMPLES_PER_CLASS = 2000                # 클래스당 목표 이미지 수 (언더샘플링 기준)
VAL_RATIO = 0.2                         # val 비율
RANDOM_SEED = 42

# 음식은 101개 카테고리 중 일부만 사용 (원하는 대로 수정 가능)
FOOD_CATEGORIES = [
    "ramen", "sushi", "pizza", "steak", "hamburger",
    "fried_rice", "spaghetti_bolognese", "ice_cream",
    "chicken_curry", "sashimi", "french_fries", "caesar_salad",
]

# 사물은 Caltech-101의 101개 카테고리 중 "일상 사물/인공물" 계열만 선택
# (동물/자연물/얼굴 카테고리는 제외 - 풍경/인물과 겹치지 않게)
OBJECT_CATEGORIES = [
    "chair", "windsor_chair", "cup", "ewer", "lamp", "laptop",
    "headphone", "watch", "scissors", "stapler", "umbrella",
    "wrench", "cellphone", "camera", "binocular", "mandolin",
]

# 인물은 Caltech-101의 Faces(배경 포함) / Faces_easy(클로즈업 크롭) 카테고리를 사용
# (단, Caltech Faces는 27명 정도만 반복 촬영돼 다양성이 낮으므로
#  ashwingupta3012/human-faces(다양한 인물 7000장+)를 주 소스로 함께 사용)
PERSON_CATEGORIES = ["Faces", "Faces_easy"]
HUMAN_FACES_CATEGORY = "Humans"

random.seed(RANDOM_SEED)


def download_datasets():
    """kagglehub로 데이터셋 다운로드 (이미 캐시돼 있으면 재다운로드 안 함)."""
    print("[1/4] Intel Image Classification 다운로드 중...")
    intel_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
    print(f"  -> {intel_path}")

    print("[1/4] Food-101 다운로드 중...")
    food_path = kagglehub.dataset_download("kmader/food41")
    print(f"  -> {food_path}")

    print("[1/4] Caltech-101 다운로드 중...")
    caltech_path = kagglehub.dataset_download("imbikramsaha/caltech-101")
    print(f"  -> {caltech_path}")

    print("[1/4] Human Faces 다운로드 중...")
    human_faces_path = kagglehub.dataset_download("ashwingupta3012/human-faces")
    print(f"  -> {human_faces_path}")

    return Path(intel_path), Path(food_path), Path(caltech_path), Path(human_faces_path)


def find_class_dir(root: Path, class_name: str):
    """다운로드된 폴더 구조가 데이터셋마다 달라서, class_name 폴더를 재귀 탐색."""
    matches = [p for p in root.rglob(class_name) if p.is_dir()]
    return matches[0] if matches else None


def collect_landscape_and_city(intel_root: Path, work_dir: Path):
    """Intel 데이터셋 -> 풍경 / 도시 로 재분류."""
    print("[2/4] 풍경/도시 폴더 재구성 중...")
    mapping = {
        "풍경": ["sea", "glacier", "mountain", "forest"],
        "도시": ["buildings", "street"],
    }
    for label, sub_classes in mapping.items():
        target_dir = work_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for sub in sub_classes:
            src_dir = find_class_dir(intel_root, sub)
            if src_dir is None:
                print(f"  ! 경고: '{sub}' 폴더를 찾지 못함 (건너뜀)")
                continue
            for img in src_dir.glob("*.jpg"):
                dst = target_dir / f"{sub}_{img.name}"
                if not dst.exists():
                    shutil.copy(img, dst)
        print(f"  -> {label}: {len(list(target_dir.glob('*')))}장")


def collect_food(food_root: Path, work_dir: Path):
    """Food-101 -> 음식 (선택된 카테고리만) 로 재분류."""
    print("[2/4] 음식 폴더 재구성 중...")
    target_dir = work_dir / "음식"
    target_dir.mkdir(parents=True, exist_ok=True)
    for cat in FOOD_CATEGORIES:
        src_dir = find_class_dir(food_root, cat)
        if src_dir is None:
            print(f"  ! 경고: 음식 카테고리 '{cat}' 를 찾지 못함 (건너뜀)")
            continue
        for img in src_dir.glob("*.jpg"):
            dst = target_dir / f"{cat}_{img.name}"
            if not dst.exists():
                shutil.copy(img, dst)
    print(f"  -> 음식: {len(list(target_dir.glob('*')))}장")


def collect_objects(caltech_root: Path, work_dir: Path):
    """Caltech-101 -> 사물 (선택된 카테고리만) 로 재분류."""
    print("[2/4] 사물 폴더 재구성 중...")
    target_dir = work_dir / "사물"
    target_dir.mkdir(parents=True, exist_ok=True)
    for cat in OBJECT_CATEGORIES:
        src_dir = find_class_dir(caltech_root, cat)
        if src_dir is None:
            print(f"  ! 경고: 사물 카테고리 '{cat}' 를 찾지 못함 (건너뜀)")
            continue
        # Caltech-101은 jpg 외에도 png 등이 섞여있을 수 있어 확장자 폭넓게 처리
        for img in list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")):
            dst = target_dir / f"{cat}_{img.name}"
            if not dst.exists():
                shutil.copy(img, dst)
    print(f"  -> 사물: {len(list(target_dir.glob('*')))}장")


def collect_persons(caltech_root: Path, human_faces_root: Path, work_dir: Path):
    """Caltech-101 Faces/Faces_easy + human-faces(Humans) -> 인물 로 재분류."""
    print("[2/4] 인물 폴더 재구성 중...")
    target_dir = work_dir / "인물"
    target_dir.mkdir(parents=True, exist_ok=True)

    for cat in PERSON_CATEGORIES:
        src_dir = find_class_dir(caltech_root, cat)
        if src_dir is None:
            print(f"  ! 경고: 인물 카테고리 '{cat}' 를 찾지 못함 (건너뜀)")
            continue
        for img in list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")):
            dst = target_dir / f"{cat}_{img.name}"
            if not dst.exists():
                shutil.copy(img, dst)

    human_dir = find_class_dir(human_faces_root, HUMAN_FACES_CATEGORY)
    if human_dir is None:
        print(f"  ! 경고: '{HUMAN_FACES_CATEGORY}' 폴더를 찾지 못함 (건너뜀)")
    else:
        for img in (list(human_dir.glob("*.jpg")) + list(human_dir.glob("*.png"))
                    + list(human_dir.glob("*.jpeg"))):
            dst = target_dir / f"human_{img.name}"
            if not dst.exists():
                shutil.copy(img, dst)

    count = len(list(target_dir.glob("*")))
    print(f"  -> 인물: {count}장")
    if count < SAMPLES_PER_CLASS:
        print(f"  ! 참고: 인물 데이터가 목표치({SAMPLES_PER_CLASS}장)보다 적음. ({count}장)")


def undersample(work_dir: Path, classes, samples_per_class: int):
    """클래스별 이미지 수를 samples_per_class로 맞춘다 (초과 시 랜덤 삭제 대상 선정)."""
    print("[3/4] 언더샘플링 중...")
    selected = {}
    for cls in classes:
        cls_dir = work_dir / cls
        images = list(cls_dir.glob("*"))
        random.shuffle(images)
        n = min(samples_per_class, len(images))
        selected[cls] = images[:n]
        print(f"  -> {cls}: {len(images)}장 중 {n}장 선택")
    return selected


def split_train_val(selected: dict, output_dir: Path, val_ratio: float):
    """train/val 폴더로 분리하여 복사."""
    print("[4/4] train/val 분리 중...")
    for cls, images in selected.items():
        n_val = int(len(images) * val_ratio)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        train_dir = output_dir / "train" / cls
        val_dir = output_dir / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir / img.name)

        print(f"  -> {cls}: train {len(train_imgs)}장 / val {len(val_imgs)}장")


def main():
    work_dir = Path("./_raw_sorted")  # 재분류 중간 결과 (임시)
    work_dir.mkdir(exist_ok=True)

    intel_root, food_root, caltech_root, human_faces_root = download_datasets()

    collect_landscape_and_city(intel_root, work_dir)
    collect_food(food_root, work_dir)
    collect_objects(caltech_root, work_dir)
    collect_persons(caltech_root, human_faces_root, work_dir)

    classes = ["풍경", "도시", "음식", "사물", "인물"]
    selected = undersample(work_dir, classes, SAMPLES_PER_CLASS)
    split_train_val(selected, OUTPUT_DIR, VAL_RATIO)

    print("\n완료! 최종 데이터셋 위치:", OUTPUT_DIR.resolve())
    print("중간 산출물(_raw_sorted)은 필요 없으면 삭제해도 됩니다.")


if __name__ == "__main__":
    main()
