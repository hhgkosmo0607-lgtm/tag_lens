"""
CNN 학습 스크립트 - MobileNetV2 fine-tuning
2-class: 실내 / 실외

사용법:
    python scripts/train/train_indoor_model.py

데이터셋 폴더 구조 (실행 전 준비, prepare_indoor_dataset.py로 자동 생성):
    dataset_indoor/
    ├── train/
    │   ├── 실내/
    │   └── 실외/
    └── val/
        ├── 실내/
        └── 실외/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset_indoor"
MODEL_PATH = BASE_DIR / "models" / "indoor_model.h5"
# 2026-07-09: 학습 중 체크포인트를 서비스 중인 MODEL_PATH에 바로 덮어쓰다가, 그 사이
# Flask 서버가 같은 파일을 읽고 있던 타이밍과 겹쳐 Windows 파일 잠금 충돌로 학습이
# 중간에 죽는 사고가 있었음 — 체크포인트는 임시 경로에 저장하고, 학습이 완전히 끝난
# 뒤에만 MODEL_PATH로 옮겨서 서비스 파일을 학습 중에는 건드리지 않도록 분리함.
TRAINING_MODEL_PATH = BASE_DIR / "models" / "indoor_model.training.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 15
EPOCHS_FINETUNE = 20
CLASS_NAMES = ["실내", "실외"]


def build_data_pipeline() -> tuple[keras.preprocessing.image.DirectoryIterator, keras.preprocessing.image.DirectoryIterator]:
    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
    )
    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_ds = train_gen.flow_from_directory(
        str(DATASET_DIR / "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_ds = val_gen.flow_from_directory(
        str(DATASET_DIR / "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_ds, val_ds


def build_model() -> tuple[keras.Model, keras.Model]:
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


def train() -> None:
    print("=" * 50)
    print("TAG_LENS - 실내/실외 CNN 학습 시작")
    print(f"클래스: {CLASS_NAMES}")
    print(f"모델 저장 경로: {MODEL_PATH}")
    print("=" * 50)

    if not (DATASET_DIR / "train").exists():
        print("[ERROR] dataset_indoor/train 폴더가 없습니다.")
        print("  → prepare_indoor_dataset.py를 먼저 실행하세요.")
        return

    TRAINING_MODEL_PATH.parent.mkdir(exist_ok=True)
    train_ds, val_ds = build_data_pipeline()

    print(f"\n훈련 샘플: {train_ds.samples}  |  검증 샘플: {val_ds.samples}")
    print(f"클래스 인덱스: {train_ds.class_indices}\n")

    # 2026-07-09: "하늘 안 보이는 실외" 보강(expand_indoor_no_sky.py)으로 실외가
    # 실내보다 훨씬 많아짐 — class_weight로 손실 함수에서 실내 비중을 높여 균형을 맞춘다.
    class_counts = {idx: 0 for idx in train_ds.class_indices.values()}
    for label in train_ds.classes:
        class_counts[label] += 1
    total = sum(class_counts.values())
    class_weight = {idx: total / (len(class_counts) * count) for idx, count in class_counts.items()}
    print(f"클래스별 샘플 수: {class_counts}")
    print(f"class_weight: {class_weight}\n")

    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n[1단계] base 동결, 분류 헤드 학습")
    callbacks_phase1 = [
        EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(TRAINING_MODEL_PATH), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=callbacks_phase1,
        class_weight=class_weight,
    )

    print("\n[2단계] fine-tuning (base 상위 30레이어 해동)")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(TRAINING_MODEL_PATH), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=callbacks_phase2,
        class_weight=class_weight,
    )

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n최종 검증 정확도: {acc:.4f}  |  손실: {loss:.4f}")

    # 학습이 끝난 뒤에만 서비스 파일을 교체 — 학습 중에는 MODEL_PATH를 건드리지 않아서
    # Flask 서버가 같은 파일을 읽고 있어도 충돌하지 않는다.
    shutil.copy2(TRAINING_MODEL_PATH, MODEL_PATH)
    TRAINING_MODEL_PATH.unlink()
    print(f"모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    train()
