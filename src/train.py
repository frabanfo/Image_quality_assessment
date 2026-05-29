"""
Training pipeline for IQA CNN regressors.

Two-phase fine-tuning:
  Phase 1 — backbone frozen, solo la testa si aggiorna, LR alto
  Phase 2 — fine-tuning completo, LR basso, early stopping su val SRCC

Interfaccia attesa dai dataset (da dataset.py dei compagni):
  train_ds, val_ds : tf.data.Dataset che emette (image, mos)
      image : tf.Tensor shape (B, 224, 224, 3), dtype float32, valori [0, 255]
      mos   : tf.Tensor shape (B,),             dtype float32
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras
from scipy.stats import spearmanr

from src.data_pipeline import prepare_datasets
from src.models import build_model_a, build_model_b, set_trainable, set_trainable_b


# ---------------------------------------------------------------------------
# Custom callback — SRCC su validation set
# ---------------------------------------------------------------------------

class SRCCCallback(keras.callbacks.Callback):
    """
    Calcola la Spearman correlation (SRCC) sul validation set a fine epoca
    e la registra nei logs come 'val_srcc'.

    Serve perché Keras non ha SRCC come metrica built-in e l'EarlyStopping
    ha bisogno di trovare 'val_srcc' nei logs per poter monitorarla.
    """

    def __init__(self, val_dataset: tf.data.Dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for images, y in self.val_dataset:
            preds = self.model(images, training=False)
            mos = y[:, 0] if y.shape.rank == 2 else y  # supporta return_std=True/False
            y_true.extend(mos.numpy())
            y_pred.extend(preds.numpy().flatten())

        srcc, _ = spearmanr(y_true, y_pred)
        logs['val_srcc'] = float(srcc)


# ---------------------------------------------------------------------------
# Fasi di training
# ---------------------------------------------------------------------------

def _phase1(model, train_ds, val_ds, epochs, lr, set_trainable_fn):
    """Backbone congelato — allena solo la testa di regressione."""
    set_trainable_fn(model, backbone_trainable=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )
    srcc_cb = SRCCCallback(val_ds)
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[srcc_cb],
        verbose=1,
    )


def _phase2(model, train_ds, val_ds, epochs, lr, patience, save_path, set_trainable_fn):
    """
    Fine-tuning progressivo in due sub-fasi:

    2a — sblocca solo i top 30 layer del backbone (ultimi ~2 blocchi),
         LR invariato, niente early stopping. Adatta le feature di alto
         livello prima di toccare i layer più profondi.

    2b — sblocca l'intero backbone con LR/10, early stopping su val_srcc.
         LR ridotto per non distruggere le feature di basso livello.
    """
    warmup_epochs = min(max(5, epochs // 4), epochs)
    remaining_epochs = epochs - warmup_epochs

    # ---- Phase 2a: top-30 layer only ----------------------------------------
    set_trainable_fn(model, backbone_trainable=True, n_top_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )
    h2a = model.fit(
        train_ds,
        epochs=warmup_epochs,
        validation_data=val_ds,
        callbacks=[SRCCCallback(val_ds)],
        verbose=1,
    )

    # ---- Phase 2b: full backbone, lower LR, early stopping ------------------
    set_trainable_fn(model, backbone_trainable=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr / 10),
        loss='mse',
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )

    callbacks = [
        SRCCCallback(val_ds),
        keras.callbacks.EarlyStopping(
            monitor='val_srcc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor='val_srcc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_srcc',
            mode='max',
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

def train(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str = 'model',
    phase1_epochs: int = 5,
    phase2_epochs: int = 30,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-5,
    patience: int = 7,
    save_dir: str = 'checkpoints',
    set_trainable_fn=set_trainable,
):
    """
    Pipeline completa di training in 2 fasi.

    Args:
        model           : modello da build_model_a() o build_model_b()
        train_ds        : tf.data.Dataset (image, mos) — training
        val_ds          : tf.data.Dataset (image, mos) — validation
        model_name      : prefisso per il file di checkpoint
        phase1_epochs   : epoche fase 1 (solo testa)
        phase2_epochs   : epoche massime fase 2 (interrotto da early stopping)
        phase1_lr       : learning rate fase 1
        phase2_lr       : learning rate fase 2
        patience        : epoche senza miglioramento su val_srcc prima dello stop
        save_dir        : cartella dove salvare il modello migliore
        set_trainable_fn: usa set_trainable per ModelA, set_trainable_b per ModelB

    Returns:
        (history_phase1, history_phase2)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_best.weights.h5")

    print(f"\n{'='*55}")
    print(f"  Phase 1 — head only  ({phase1_epochs} epochs, lr={phase1_lr})")
    print(f"{'='*55}")
    h1 = _phase1(model, train_ds, val_ds, phase1_epochs, phase1_lr, set_trainable_fn)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — full fine-tuning  (max {phase2_epochs} epochs, lr={phase2_lr})")
    print(f"  Early stopping: patience={patience} su val_srcc")
    print(f"{'='*55}")
    h2a, h2b = _phase2(model, train_ds, val_ds, phase2_epochs, phase2_lr, patience,
                       save_path, set_trainable_fn)

    print(f"\nModello migliore salvato in: {save_path}")
    return h1, h2a, h2b


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_training_target(model_name):
    if model_name == "a":
        return build_model_a(), "model_a", set_trainable

    return build_model_b(), "model_b", set_trainable_b


# ---------------------------------------------------------------------------
# Entry point per training / smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size,
        save_csv=True,
    )

    model, model_name, set_trainable_fn = build_training_target(args.model)
    model.summary()

    if args.smoke_test:
        for images, mos in train_ds.take(1):
            preds = model(images, training=False)
            print("\nSmoke test pipeline + model:")
            print("Images shape:", images.shape)
            print("Images range:", float(tf.reduce_min(images)), float(tf.reduce_max(images)))
            print("MOS shape:", mos.shape)
            print("Predictions shape:", preds.shape)
    else:
        train(
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
        )
