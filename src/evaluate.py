"""
Evaluate IQA models on the test split.

Produces:
  - Global metrics (PLCC, SRCC, RMSE, MAE, MSE)
  - Metrics by MOS range (low / medium / high quality)
  - Metrics by observer StdDev (low / medium / high ambiguity)
  - Scatter plot: predicted vs real MOS
  - Scatter plot: absolute error vs observer StdDev
  - Bland-Altman plot
  - Bar charts of SRCC/RMSE across subgroups
  - Grad-CAM visualizations (models A and B; requires opencv-python)

Examples:
    python -m src.evaluate --model a
    python -m src.evaluate --model b --checkpoint checkpoints/model_b_best.weights.h5
    python -m src.evaluate --model a --output-dir results/model_a --no-gradcam
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

# Swin checkpoints are trained and saved under tf-keras (Keras 2, see train_vit.py),
# whose .weights.h5 layout is incompatible with Keras 3. The env var must be set
# before `import tensorflow`, hence the argv sniff instead of argparse.
if any(arg in ("swin", "vit") for arg in sys.argv):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Windows console/pipes default to cp1252, which can't encode the box-drawing
# characters used in the report tables.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from tensorflow import keras

from src.data_pipeline import BATCH_SIZE, prepare_datasets

MOS_MIN = 0.0
MOS_MAX = 100.0


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_or_load_model(
    model_name: str,
    checkpoint: str | None = None,
    img_size: int = 224,
    backbone_name: str | None = None,
) -> keras.Model:
    from src.models import build_model_a, build_model_b

    input_shape = (img_size, img_size, 3)
    if model_name == "a":
        model = build_model_a(input_shape=input_shape)
    elif model_name == "v2s":
        model = build_model_a(input_shape=input_shape, backbone="v2s")
    elif model_name == "b":
        model = build_model_b(input_shape=input_shape)
    elif model_name == "bv2s":
        model = build_model_b(input_shape=input_shape, backbone="v2s")
    elif model_name in ("swin", "vit"):
        from src.models_vit_pretrained import build_model_vit, DEFAULT_BACKBONE
        model = build_model_vit(backbone_name=backbone_name or DEFAULT_BACKBONE)
        model(tf.zeros((1, img_size, img_size, 3), dtype=tf.float32), training=False)
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Use 'a', 'b', 'v2s', or 'swin'.")

    if checkpoint:
        if checkpoint.endswith(".weights.h5"):
            model.load_weights(checkpoint)
        else:
            model = keras.models.load_model(checkpoint)

    return model


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


def collect_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) normalized in [0, 1], preserving dataset order."""
    y_true, y_pred = [], []
    for i, (images, mos) in enumerate(dataset):
        if max_batches is not None and i >= max_batches:
            break
        preds = model(images, training=False)
        y_true.extend(mos.numpy().reshape(-1))
        y_pred.extend(preds.numpy().reshape(-1))
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(np.mean(np.square(y_true - y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)
    return {"plcc": float(plcc), "srcc": float(srcc), "rmse": rmse, "mae": mae, "mse": mse}


def _print_metrics(metrics: dict, title: str = "Metrics") -> None:
    print(f"\n{title}")
    print("-" * max(len(title), 30))
    for name, value in metrics.items():
        print(f"  {name.upper():<6}: {value:.4f}")


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------


def analyze_by_mos_range(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Use quartile-based thresholds so each bucket has ~25% of samples,
    # matching the stratification used during data splitting.
    p25, p75 = np.percentile(y_true, [25, 75])
    ranges = {
        f"low  (< {p25:.1f})":          y_true < p25,
        f"med  ({p25:.1f}-{p75:.1f})":  (y_true >= p25) & (y_true < p75),
        f"high (> {p75:.1f})":           y_true >= p75,
    }
    results = {}
    for label, mask in ranges.items():
        if mask.sum() < 5:
            continue
        results[label] = {**compute_metrics(y_true[mask], y_pred[mask]), "n": int(mask.sum())}
    return results


def analyze_by_stddev(y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray) -> dict:
    p33, p67 = np.percentile(std, [33, 67])
    ranges = {
        f"low  std (< {p33:.1f})": std < p33,
        f"med  std ({p33:.1f}-{p67:.1f})": (std >= p33) & (std < p67),
        f"high std (> {p67:.1f})": std >= p67,
    }
    results = {}
    for label, mask in ranges.items():
        if mask.sum() < 5:
            continue
        results[label] = {**compute_metrics(y_true[mask], y_pred[mask]), "n": int(mask.sum())}
    return results


def print_subgroup_table(results: dict, title: str) -> None:
    print(f"\n{title}")
    print("─" * 74)
    print(f"  {'Range':<28} {'N':>5}  {'PLCC':>6}  {'SRCC':>6}  {'RMSE':>7}  {'MAE':>7}")
    print("─" * 74)
    for label, m in results.items():
        print(
            f"  {label:<28} {m['n']:>5}  {m['plcc']:>6.3f}  {m['srcc']:>6.3f}"
            f"  {m['rmse']:>7.3f}  {m['mae']:>7.3f}"
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str) -> None:
    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=10, color="steelblue", label="samples")
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real MOS", fontsize=11)
    ax.set_ylabel("Predicted MOS", fontsize=11)
    ax.set_title(f"{title}\nSRCC={srcc:.3f}  PLCC={plcc:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    _savefig(fig, save_path)


def plot_error_vs_std(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    abs_err = np.abs(y_true - y_pred)
    corr, _ = pearsonr(std, abs_err)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(std, abs_err, alpha=0.35, s=10, color="darkorange")
    z = np.polyfit(std, abs_err, 1)
    xline = np.linspace(std.min(), std.max(), 200)
    ax.plot(xline, np.poly1d(z)(xline), "r-", lw=1.5, label=f"trend  r={corr:.3f}")
    ax.set_xlabel("Observer StdDev", fontsize=11)
    ax.set_ylabel("Absolute Error (MOS)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    _savefig(fig, save_path)


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    mean_val = (y_true + y_pred) / 2
    diff = y_pred - y_true
    bias = diff.mean()
    loa = 1.96 * diff.std()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean_val, diff, alpha=0.35, s=10, color="mediumseagreen")
    ax.axhline(bias, color="red", lw=1.5, label=f"bias = {bias:+.2f}")
    ax.axhline(bias + loa, color="gray", lw=1.0, ls="--", label=f"±1.96σ = {loa:.2f}")
    ax.axhline(bias - loa, color="gray", lw=1.0, ls="--")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Mean of Real and Predicted MOS", fontsize=11)
    ax.set_ylabel("Predicted − Real MOS", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    _savefig(fig, save_path)


def plot_bar_metric(results: dict, metric: str, save_path: str, title: str) -> None:
    labels = list(results.keys())
    values = [results[k][metric] for k in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2), 4))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelrotation=10)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# Grad-CAM (CNN models only)
# ---------------------------------------------------------------------------


def _build_gradcam_model(model: keras.Model, model_type: str) -> keras.Model:
    """
    Dual-output model returning (conv_features, prediction) for Grad-CAM.

    Model A / V2S: last spatial tensor before GAP is gap.input  (B, 7, 7, 1280)
    Model B: top-scale branch before GAP is gap_top.input       (B, 7, 7, 1280)
    """
    if model_type in ("a", "v2s"):
        conv_tensor = model.get_layer("gap").input
    elif model_type in ("b", "bv2s"):
        conv_tensor = model.get_layer("gap_top").input
    else:
        raise ValueError(f"Grad-CAM not supported for model type '{model_type}'")
    return keras.Model(inputs=model.inputs, outputs=[conv_tensor, model.output])


def _gradcam_heatmap(gradcam_model: keras.Model, img: np.ndarray) -> np.ndarray:
    img_t = tf.cast(img[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_t, training=False)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)  # (1, H, W, C)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def _overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    jet = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    return np.uint8(img_u8 * (1 - alpha) + jet * alpha)


def visualize_gradcam(
    model: keras.Model,
    model_type: str,
    test_df: pd.DataFrame,
    save_dir: str,
    n_per_group: int = 2,
    img_size: int = 224,
) -> None:
    """
    Grad-CAM on 4 representative groups:
      high MOS + low std,  high MOS + high std,
      low MOS  + low std,  low MOS  + high std.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("  Grad-CAM skipped: install opencv-python to enable it.")
        return

    try:
        gradcam_model = _build_gradcam_model(model, model_type)
    except ValueError as exc:
        print(f"  Grad-CAM skipped: {exc}")
        return

    mos_med = test_df["mos"].median()
    std_med = test_df["std"].median()
    groups = {
        "high_mos_low_std": test_df[
            (test_df["mos"] >= mos_med) & (test_df["std"] <= std_med)
        ],
        "high_mos_high_std": test_df[
            (test_df["mos"] >= mos_med) & (test_df["std"] > std_med)
        ],
        "low_mos_low_std": test_df[
            (test_df["mos"] < mos_med) & (test_df["std"] <= std_med)
        ],
        "low_mos_high_std": test_df[
            (test_df["mos"] < mos_med) & (test_df["std"] > std_med)
        ],
    }

    out_dir = os.path.join(save_dir, "gradcam")
    for group_name, group_df in groups.items():
        sample = group_df.sample(min(n_per_group, len(group_df)), random_state=42)
        for i, (_, row) in enumerate(sample.iterrows()):
            raw = tf.io.read_file(row["image_path"])
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, [img_size, img_size]).numpy()

            heatmap = _gradcam_heatmap(gradcam_model, img)
            overlay = _overlay_heatmap(img, heatmap)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img.astype(np.uint8))
            axes[0].set_title(f"MOS={row['mos']:.1f}  std={row['std']:.1f}", fontsize=9)
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM", fontsize=9)
            axes[1].axis("off")
            fig.suptitle(group_name.replace("_", " "), fontsize=10)
            _savefig(fig, os.path.join(out_dir, f"{group_name}_{i}.png"))


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    model: keras.Model,
    model_type: str,
    test_ds: tf.data.Dataset,
    test_df: pd.DataFrame,
    output_dir: str,
    label: str,
    max_batches: int | None = None,
    gradcam: bool = True,
    img_size: int = 224,
) -> dict:
    """
    Run the full evaluation suite and save all plots under output_dir.
    Returns global metrics dict (MOS scale [0, 100]).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nCollecting predictions...")
    y_true_n, y_pred_n = collect_predictions(model, test_ds, max_batches)

    # Rescale from [0, 1] to MOS scale [0, 100]
    y_true = y_true_n * (MOS_MAX - MOS_MIN)
    y_pred = y_pred_n * (MOS_MAX - MOS_MIN)
    # test_ds preserves insertion order (no shuffle) → std aligns by index
    std = test_df["std"].values[: len(y_true)]

    print(f"  Samples: {len(y_true)}")

    global_metrics = compute_metrics(y_true, y_pred)
    _print_metrics(global_metrics, f"Global metrics — {label}")

    mos_results = analyze_by_mos_range(y_true, y_pred)
    print_subgroup_table(mos_results, "Metrics by MOS range")

    std_results = analyze_by_stddev(y_true, y_pred, std)
    print_subgroup_table(std_results, "Metrics by observer StdDev")

    print("\nGenerating plots...")
    plot_scatter(
        y_true, y_pred,
        os.path.join(output_dir, "scatter_mos.png"),
        f"{label} — Predicted vs Real MOS",
    )
    plot_error_vs_std(
        y_true, y_pred, std,
        os.path.join(output_dir, "error_vs_std.png"),
        f"{label} — Prediction Error vs Ambiguity",
    )
    plot_bland_altman(
        y_true, y_pred,
        os.path.join(output_dir, "bland_altman.png"),
        f"{label} — Bland-Altman",
    )
    if mos_results:
        plot_bar_metric(
            mos_results, "srcc",
            os.path.join(output_dir, "srcc_by_mos_range.png"),
            f"{label} — SRCC by MOS range",
        )
        plot_bar_metric(
            mos_results, "rmse",
            os.path.join(output_dir, "rmse_by_mos_range.png"),
            f"{label} — RMSE by MOS range",
        )
    if std_results:
        plot_bar_metric(
            std_results, "srcc",
            os.path.join(output_dir, "srcc_by_std_range.png"),
            f"{label} — SRCC by StdDev range",
        )
        plot_bar_metric(
            std_results, "rmse",
            os.path.join(output_dir, "rmse_by_std_range.png"),
            f"{label} — RMSE by StdDev range",
        )

    if gradcam:
        print("\nGenerating Grad-CAM visualizations...")
        visualize_gradcam(model, model_type, test_df, output_dir, img_size=img_size)

    return global_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an IQA model on the test split.")
    parser.add_argument(
        "--model",
        choices=["a", "b", "v2s", "bv2s", "swin"],
        default="a",
        help="Architecture: a=baseline CNN, b=multi-scale CNN, v2s=EfficientNetV2-S, "
             "bv2s=multi-scale V2S, swin=Swin-Tiny",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .weights.h5 or saved .keras model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Process only the first N batches (quick test)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for plots (default: results/model_<x>)")
    parser.add_argument("--split-dir", default="Splits",
                        help="Directory containing the split CSVs")
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip Grad-CAM visualizations")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Risoluzione di input: deve combaciare con quella di training")
    parser.add_argument("--backbone-name", default=None,
                        help="Solo per --model swin: nome backbone HuggingFace "
                             "(es. microsoft/swin-base-patch4-window12-384)")
    parser.add_argument("--include-val", action="store_true",
                        help="Evaluate on val+test combined (more stable estimate, "
                             "slightly optimistic: val was used for model selection)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = {"a": "Model A", "b": "Model B", "v2s": "Model A (EfficientNetV2-S)",
                 "bv2s": "Model B (EfficientNetV2-S multiscale)", "swin": "Model Swin-Tiny"}
    label = label_map.get(args.model, f"Model {args.model.upper()}")
    if args.model in ("swin", "vit") and args.backbone_name:
        if "base" in args.backbone_name:
            label = "Model Swin-Base"
        elif "tiny" in args.backbone_name:
            label = "Model Swin-Tiny"
    suffix = "_valtest" if args.include_val else ""
    output_dir = args.output_dir or os.path.join("results", f"model_{args.model}{suffix}")

    # save_csv=True ensures Splits/test.csv exists for std alignment
    _, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size, save_csv=True, img_size=args.img_size)

    test_csv = os.path.join(args.split_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Test split not found at '{test_csv}'. "
            "Run prepare_datasets(save_csv=True) first."
        )
    test_df = pd.read_csv(test_csv)

    if args.include_val:
        # val_ds and test_ds are both unshuffled and ordered like their CSVs,
        # so concatenation keeps predictions aligned with the stacked dataframe.
        val_df = pd.read_csv(os.path.join(args.split_dir, "val.csv"))
        test_df = pd.concat([val_df, test_df], ignore_index=True)
        test_ds = val_ds.concatenate(test_ds)
        label = f"{label} (val+test)"

    model = build_or_load_model(args.model, args.checkpoint, img_size=args.img_size,
                                backbone_name=args.backbone_name)

    run_evaluation(
        model=model,
        model_type=args.model,
        test_ds=test_ds,
        test_df=test_df,
        output_dir=output_dir,
        label=label,
        max_batches=args.max_batches,
        gradcam=not args.no_gradcam and args.model != "swin",
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
