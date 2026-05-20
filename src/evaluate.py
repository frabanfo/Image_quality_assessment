"""
Evaluate IQA models on the test split.

Examples:
    python -m src.evaluate --model a
    python -m src.evaluate --model b
    python -m src.evaluate --checkpoint checkpoints/model_a_best.keras
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from tensorflow import keras

from src.data_pipeline import BATCH_SIZE, prepare_datasets
from src.models import build_model_a, build_model_b


def collect_predictions(model: keras.Model, dataset: tf.data.Dataset, max_batches=None):
    y_true = []
    y_pred = []

    for batch_idx, (images, mos) in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break

        preds = model(images, training=False)
        y_true.extend(mos.numpy().reshape(-1))
        y_pred.extend(preds.numpy().reshape(-1))

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def compute_metrics(y_true, y_pred):
    mse = float(np.mean(np.square(y_true - y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "srcc": float(srcc),
        "plcc": float(plcc),
    }


def build_or_load_model(model_name, checkpoint=None):
    if model_name == "a":
        model = build_model_a()
    elif model_name == "b":
        model = build_model_b()
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

    if checkpoint:
        if checkpoint.endswith(".weights.h5"):
            model.load_weights(checkpoint)
        else:
            model = keras.models.load_model(checkpoint)

    return model


def print_metrics(metrics):
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, test_ds = prepare_datasets(
        batch_size=args.batch_size,
        save_csv=False,
    )
    model = build_or_load_model(args.model, args.checkpoint)

    y_true, y_pred = collect_predictions(
        model,
        test_ds,
        max_batches=args.max_batches,
    )
    metrics = compute_metrics(y_true, y_pred)

    print(f"\nSamples evaluated: {len(y_true)}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
