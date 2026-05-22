"""
Dedicated training script for the pretrained ViT regressor.

Run from the project root:
    python3 -m src.train_vit --phase1-epochs 3 --phase2-epochs 10
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

from scipy.stats import pearsonr, spearmanr

from src.data_pipeline import prepare_datasets
from src.models_vit_pretrained import (
    DEFAULT_BACKBONE,
    build_model_vit,
    set_trainable_vit,
)
from src.train import train


MODEL_NAME = "model_vit_pretrained"
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 12
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PATIENCE = 5
SAVE_DIR = "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pretrained ViT IQA regressor.")
    parser.add_argument("--backbone-name", default=DEFAULT_BACKBONE, help="Hugging Face model id.")
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--phase1-lr", type=float, default=PHASE1_LR)
    parser.add_argument("--phase2-lr", type=float, default=PHASE2_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-csv", action="store_true")
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

    model_vit = build_model_vit(backbone_name=args.backbone_name)
    model_vit.summary()

    train(
        model=model_vit,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=args.model_name,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        patience=args.patience,
        save_dir=args.save_dir,
        set_trainable_fn=set_trainable_vit,
    )

    evaluate_on_test(model_vit, test_ds)


if __name__ == "__main__":
    main(parse_args())
