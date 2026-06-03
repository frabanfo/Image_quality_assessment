"""
Dedicated training script for the pretrained Swin-Tiny regressor.

Run from the project root:
    python3 -m src.train_vit --phase1-epochs 3 --phase2-epochs 10
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

from scipy.stats import pearsonr, spearmanr
import tensorflow as tf

from src.data_pipeline import prepare_datasets
from src.models_vit_pretrained import (
    DEFAULT_BACKBONE,
    build_model_vit,
    set_trainable_vit,
)
from src.train import train


MODEL_NAME = "model_swin_tiny"
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 30
PHASE1_LR = 1e-3
PHASE2_LR = 5e-6
PATIENCE = 7
SAVE_DIR = "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pretrained Swin-Tiny IQA regressor.")
    parser.add_argument("--backbone-name", default=DEFAULT_BACKBONE, help="Hugging Face model id.")
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--phase1-lr", type=float, default=PHASE1_LR)
    parser.add_argument("--phase2-lr", type=float, default=PHASE2_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--disable-model-augmentation",
        action="store_true",
        help="Disable the local augmentation applied inside the Swin model.",
    )
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=0,
        help="Random patches to extract per image (0 = disabled). When active, also pass --disable-model-augmentation.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=192,
        help="Patch size in pixels for patch sampling (default: 192).",
    )
    return parser.parse_args()


def evaluate_on_test(model_vit, test_ds):
    y_true, y_pred = [], []

    for images, mos in test_ds:
        preds = model_vit(images, training=False)
        y_true.extend(mos.numpy())
        y_pred.extend(preds.numpy().flatten())

    mse = tf.reduce_mean(tf.square(tf.constant(y_true) - tf.constant(y_pred)))
    mae = tf.reduce_mean(tf.abs(tf.constant(y_true) - tf.constant(y_pred)))
    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)

    print("\nTest set evaluation:")
    print(f"MSE : {float(mse):.4f}")
    print(f"MAE : {float(mae):.4f}")
    print(f"SRCC: {float(srcc):.4f}")
    print(f"PLCC: {float(plcc):.4f}")


def main(args):
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size,
        save_csv=args.save_csv,
        use_augmentation=False,
        use_patch_sampling=args.patches_per_image > 0,
        patches_per_image=args.patches_per_image,
        patch_size=args.patch_size,
    )

    model_vit = build_model_vit(
        backbone_name=args.backbone_name,
        use_model_augmentation=not args.disable_model_augmentation,
    )
    model_vit(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=True)
    model_vit.summary()

    if args.smoke_test:
        for images, mos in train_ds.take(1):
            preds = model_vit(images, training=False)
            print("\nSmoke test Swin pipeline:")
            print("Images shape:", images.shape)
            print("MOS shape   :", mos.shape)
            print("Preds shape :", preds.shape)
        return

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
        phase2a_n_top_layers=2,
    )

    evaluate_on_test(model_vit, test_ds)


if __name__ == "__main__":
    main(parse_args())
