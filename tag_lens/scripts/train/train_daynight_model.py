"""
CNN 학습 스크립트 - MobileNetV2 fine-tuning
2-class: 주간 / 야간

사용법:
    python scripts/train/train_daynight_model.py

데이터셋 폴더 구조 (실행 전 준비, prepare_daynight_dataset.py로 자동 생성):
    dataset_daynight/
    ├── train/
    │   ├── 주간/
    │   └── 야간/
    └── val/
        ├── 주간/
        └── 야간/
"""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset_daynight"
MODEL_PATH = BASE_DIR / "models" / "daynight_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # 데이터 규모가 작아 배치 크기도 작게
EPOCHS_FROZEN = 15
EPOCHS_FINETUNE = 20
CLASS_NAMES = ["주간", "야간"]


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
    print("TAG_LENS - 주간/야간 CNN 학습 시작")
    print(f"클래스: {CLASS_NAMES}")
    print(f"모델 저장 경로: {MODEL_PATH}")
    print("=" * 50)

    if not (DATASET_DIR / "train").exists():
        print("[ERROR] dataset_daynight/train 폴더가 없습니다.")
        print("  → prepare_daynight_dataset.py를 먼저 실행하세요.")
        return

    MODEL_PATH.parent.mkdir(exist_ok=True)
    train_ds, val_ds = build_data_pipeline()

    print(f"\n훈련 샘플: {train_ds.samples}  |  검증 샘플: {val_ds.samples}")
    print(f"클래스 인덱스: {train_ds.class_indices}\n")

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
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=callbacks_phase1,
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
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=callbacks_phase2,
    )

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n최종 검증 정확도: {acc:.4f}  |  손실: {loss:.4f}")
    print(f"모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    train()
