"""
Dedicated training script for the ViT IQA regressor.

Run from the project root:
    python3 -m src.train_vit
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

from scipy.stats import pearsonr, spearmanr
from src.data_pipeline import prepare_datasets
from tensorflow import keras

from src.models import build_model_vit
from src.train import SRCCCallback


MODEL_NAME = "model_vit"
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
SAVE_DIR = "checkpoints"


def evaluate_on_test(model_vit, test_ds):
    y_true, y_pred = [], []

    for images, mos in test_ds:
        preds = model_vit(images, training=False)
        y_true.extend(mos.numpy())
        y_pred.extend(preds.numpy().flatten())

    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)

    print("\nTest set evaluation:")
    print(f"SRCC: {float(srcc):.4f}")
    print(f"PLCC: {float(plcc):.4f}")


def main():
    train_ds, val_ds, test_ds = prepare_datasets(save_csv=True)

    model_vit = build_model_vit()
    model_vit.summary()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.keras")

    model_vit.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    callbacks = [
        SRCCCallback(val_ds),
        keras.callbacks.EarlyStopping(
            monitor="val_srcc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_srcc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model_vit.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nModello migliore salvato in: {save_path}")
    evaluate_on_test(model_vit, test_ds)


if __name__ == "__main__":
    main()
