"""
Training pipeline con Weighted MSE — peso inversamente proporzionale alla SD.

y_true shape: (B, 2) dove y_true[:, 0] = mos_norm, y_true[:, 1] = std_norm.
Il dataset va preparato con prepare_datasets(return_std=True).

La loss penalizza di più gli errori su immagini con bassa SD (annotatori
in accordo → label affidabile) e di meno quelli con alta SD (label rumorosa).

  w_i = 1 / (std_norm_i + eps)
  w_i <- w_i / mean(w_i)   # normalizzazione: loss su stessa scala di MSE
  L    = mean(w_i * (mos_i - pred_i)^2)

Stessa architettura e pipeline a 2 fasi di train.py — i checkpoint vengono
salvati separatamente (suffisso _wmse) per confronto diretto.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras

from src.data_pipeline import prepare_datasets
from src.models import build_model_a, build_model_b, set_trainable, set_trainable_b
from src.train import SRCCCallback


# ---------------------------------------------------------------------------
# Loss pesata sulla SD
# ---------------------------------------------------------------------------

class WeightedMSELoss(keras.losses.Loss):
    """
    MSE pesata esponenzialmente sulla deviazione standard delle annotazioni.

    y_true: tensor (B, 2) — colonna 0 = mos_norm, colonna 1 = std_norm.
    y_pred: tensor (B, 1) — previsione del modello.

    Formula:  w = exp(-alpha * std_norm)
    - alpha=0  → tutti i pesi uguali (MSE classica)
    - alpha>0  → bassa SD = peso alto, alta SD = peso basso
    - La curva esponenziale è più smooth dell'inversa: evita pesi estremi
      su campioni con std ≈ 0 e differenzia meglio la zona intermedia.

    I pesi vengono normalizzati a media=1 per mantenere la loss sulla
    stessa scala della MSE standard.
    """

    def __init__(self, alpha: float = 1.0, name: str = "weighted_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        mos = y_true[:, 0:1]
        std = y_true[:, 1:2]

        weights = tf.exp(-self.alpha * std)
        weights = weights / tf.reduce_mean(weights)  # normalizza a media=1

        return tf.reduce_mean(weights * tf.square(mos - y_pred))

    def get_config(self):
        cfg = super().get_config()
        cfg["alpha"] = self.alpha
        return cfg


# ---------------------------------------------------------------------------
# Fasi di training
# ---------------------------------------------------------------------------

def _phase1(model, train_ds, val_ds, epochs, lr, set_trainable_fn, loss_fn):
    set_trainable_fn(model, backbone_trainable=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
    )
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[SRCCCallback(val_ds)],
        verbose=1,
    )


def _phase2(model, train_ds, val_ds, epochs, lr, patience, save_path,
            set_trainable_fn, loss_fn):
    warmup_epochs = max(5, epochs // 4)
    remaining_epochs = epochs - warmup_epochs

    # Phase 2a: top-30 layer only
    set_trainable_fn(model, backbone_trainable=True, n_top_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
    )
    h2a = model.fit(
        train_ds,
        epochs=warmup_epochs,
        validation_data=val_ds,
        callbacks=[SRCCCallback(val_ds)],
        verbose=1,
    )

    # Phase 2b: full backbone, lower LR, early stopping
    set_trainable_fn(model, backbone_trainable=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr / 10),
        loss=loss_fn,
    )

    callbacks = [
        SRCCCallback(val_ds),
        keras.callbacks.EarlyStopping(
            monitor="val_srcc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_srcc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_srcc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    h2b = model.fit(
        train_ds,
        epochs=remaining_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    return h2a, h2b


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def train_weighted(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str = "model",
    phase1_epochs: int = 5,
    phase2_epochs: int = 30,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-5,
    patience: int = 7,
    save_dir: str = "checkpoints",
    set_trainable_fn=set_trainable,
    alpha: float = 1.0,
):
    """
    Stessa pipeline a 2 fasi di train.py ma con WeightedMSELoss.

    Il checkpoint ha il suffisso _wmse per evitare sovrascrittura dei
    modelli addestrati con MSE standard.

    train_ds / val_ds devono essere preparati con return_std=True.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_wmse_best.weights.h5")
    loss_fn = WeightedMSELoss(alpha=alpha)

    print(f"\n{'='*60}")
    print(f"  WeightedMSE training — alpha={alpha}")
    print(f"{'='*60}")

    print(f"\n{'='*55}")
    print(f"  Phase 1 — head only  ({phase1_epochs} epochs, lr={phase1_lr})")
    print(f"{'='*55}")
    h1 = _phase1(model, train_ds, val_ds, phase1_epochs, phase1_lr,
                 set_trainable_fn, loss_fn)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — full fine-tuning  (max {phase2_epochs} epochs, lr={phase2_lr})")
    print(f"  Early stopping: patience={patience} su val_srcc")
    print(f"{'='*55}")
    h2a, h2b = _phase2(model, train_ds, val_ds, phase2_epochs, phase2_lr, patience,
                       save_path, set_trainable_fn, loss_fn)

    print(f"\nModello migliore salvato in: {save_path}")
    return h1, h2a, h2b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train IQA model with uncertainty-weighted MSE loss."
    )
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Pendenza nella loss: w = exp(-alpha * std_norm)")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_training_target(model_name):
    if model_name == "a":
        return build_model_a(), "model_a", set_trainable
    return build_model_b(), "model_b", set_trainable_b


if __name__ == "__main__":
    args = parse_args()
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size,
        save_csv=True,
        return_std=True,
    )

    model, model_name, set_trainable_fn = build_training_target(args.model)
    model.summary()

    if args.smoke_test:
        for images, y in train_ds.take(1):
            preds = model(images, training=False)
            print("\nSmoke test WeightedMSE pipeline:")
            print("Images shape:", images.shape)
            print("y shape (mos+std):", y.shape)
            print("mos_norm range:", float(tf.reduce_min(y[:, 0])),
                  float(tf.reduce_max(y[:, 0])))
            print("std_norm range:", float(tf.reduce_min(y[:, 1])),
                  float(tf.reduce_max(y[:, 1])))
            print("Predictions shape:", preds.shape)
            loss_fn = WeightedMSELoss(alpha=args.alpha)
            loss_val = loss_fn(y, preds)
            print("Loss on first batch:", float(loss_val))
    else:
        train_weighted(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            model_name=model_name,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_lr=args.phase1_lr,
            phase2_lr=args.phase2_lr,
            patience=args.patience,
            save_dir=args.save_dir,
            set_trainable_fn=set_trainable_fn,
            alpha=args.alpha,
        )
