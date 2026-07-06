"""
CNN 학습 스크립트 - 2단계(fine-tuning)만 독립 실행

train_model.py의 1단계(base 동결)가 이미 끝나서 models/genre_model.h5에
체크포인트가 저장돼 있다는 전제 하에, 그 체크포인트를 불러와
상위 30레이어를 해동하고 이어서 fine-tuning한다.
1단계를 다시 돌리지 않아도 되므로 세션을 나눠서 재개할 때 사용한다.

사용법:
    python train_model_phase2.py
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from train_model import EPOCHS_FINETUNE, MODEL_PATH, build_data_pipeline

BASE_LAYER_NAME = "mobilenetv2_1.00_224"


def train_phase2() -> None:
    print("=" * 50)
    print("TAG_LENS - CNN 2단계(fine-tuning) 재시작")
    print(f"체크포인트: {MODEL_PATH}")
    print("=" * 50)

    if not MODEL_PATH.exists():
        print(f"[ERROR] {MODEL_PATH} 가 없습니다. 1단계를 먼저 끝내세요.")
        return

    train_ds, val_ds = build_data_pipeline()
    print(f"\n훈련 샘플: {train_ds.samples}  |  검증 샘플: {val_ds.samples}\n")

    model = keras.models.load_model(str(MODEL_PATH))

    base = model.get_layer(BASE_LAYER_NAME)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n[2단계] fine-tuning (base 상위 30레이어 해동)")
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
    train_phase2()
