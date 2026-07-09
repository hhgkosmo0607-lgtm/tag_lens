"""
TAG_LENS - 실내/실외 CNN 2단계(fine-tuning) 학습 재개 (일회성 복구 스크립트)

2026-07-09: train_indoor_model.py 실행 중 Flask 서버와 체크포인트 파일 접근이
겹쳐 Windows 파일 잠금 충돌로 2단계 4에폭째에 학습이 죽은 사고가 있었음(원인은
train_indoor_model.py에서 이미 수정함 — 체크포인트를 임시 파일에 저장 후 완료 시에만
서비스 파일로 교체). 죽기 전 저장된 models/indoor_model.h5(1단계 완료 + 2단계 3에폭
완료, val_loss 0.08243)는 멀쩡하므로, 1단계부터 다시 돌리지 않고 이 체크포인트에서
2단계만 이어서 진행한다.

사용법:
    python scripts/train/resume_indoor_training.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from train_indoor_model import (
    DATASET_DIR,
    EPOCHS_FINETUNE,
    MODEL_PATH,
    TRAINING_MODEL_PATH,
    build_data_pipeline,
)

RESUME_FROM_EPOCH = 3  # 2단계 3에폭까지 완료된 상태에서 이어감


def resume() -> None:
    print("=" * 50)
    print("TAG_LENS - 실내/실외 CNN 2단계 학습 재개")
    print(f"체크포인트: {MODEL_PATH} (2단계 {RESUME_FROM_EPOCH}에폭 완료 상태)")
    print("=" * 50)

    if not (DATASET_DIR / "train").exists():
        print("[ERROR] dataset_indoor/train 폴더가 없습니다.")
        return

    TRAINING_MODEL_PATH.parent.mkdir(exist_ok=True)
    train_ds, val_ds = build_data_pipeline()

    print(f"\n훈련 샘플: {train_ds.samples}  |  검증 샘플: {val_ds.samples}")

    class_counts = {idx: 0 for idx in train_ds.class_indices.values()}
    for label in train_ds.classes:
        class_counts[label] += 1
    total = sum(class_counts.values())
    class_weight = {idx: total / (len(class_counts) * count) for idx, count in class_counts.items()}
    print(f"class_weight: {class_weight}\n")

    model = keras.models.load_model(str(MODEL_PATH))
    # 2단계 학습률(1e-5)로 재컴파일 — 옵티마이저 모멘텀 상태는 초기화되지만
    # (레이어 동결 상태는 저장된 모델에 그대로 남아있어 영향 적음), 나머지 학습에
    # 큰 영향은 없음
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(TRAINING_MODEL_PATH), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        initial_epoch=RESUME_FROM_EPOCH,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n최종 검증 정확도: {acc:.4f}  |  손실: {loss:.4f}")

    if TRAINING_MODEL_PATH.exists():
        shutil.copy2(TRAINING_MODEL_PATH, MODEL_PATH)
        TRAINING_MODEL_PATH.unlink()
        print(f"모델 저장 완료: {MODEL_PATH}")
    else:
        print("개선된 체크포인트가 없어 기존 모델을 그대로 둠(이미 최선 상태였음).")


if __name__ == "__main__":
    resume()
