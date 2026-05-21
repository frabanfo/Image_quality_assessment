"""
Dedicated training script for the ViT IQA regressor.

Run from the project root:
    python3 -m src.train_vit --epochs 20 --learning-rate 1e-4
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ViT IQA regressor.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Early stopping patience on validation SRCC.",
    )
    parser.add_argument(
        "--save-dir",
        default=SAVE_DIR,
        help="Directory where the best checkpoint will be stored.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Prefix used for the saved checkpoint filename.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size passed to prepare_datasets().",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save train/val/test CSV splits to disk.",
    )
    return parser.parse_args()


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


def main(args):
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size,
        save_csv=args.save_csv,
    )

    model_vit = build_model_vit()
    model_vit.summary()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}_best.keras")

    model_vit.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
    )

    callbacks = [
        SRCCCallback(val_ds),
        keras.callbacks.EarlyStopping(
            monitor="val_srcc",
            mode="max",
            patience=args.patience,
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
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nModello migliore salvato in: {save_path}")
    evaluate_on_test(model_vit, test_ds)


if __name__ == "__main__":
    main(parse_args())
