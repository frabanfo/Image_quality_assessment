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

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import numpy as np
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
        for images, mos in self.val_dataset:
            preds = self.model(images, training=False)
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
    """Fine-tuning completo con early stopping su val SRCC."""
    set_trainable_fn(model, backbone_trainable=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
    )

    callbacks = [
        SRCCCallback(val_ds),
        keras.callbacks.EarlyStopping(
            monitor='val_srcc',
            mode='max',
            patience=patience,
            restore_best_weights=True,  # ricarica i pesi dell'epoca migliore
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor='val_srcc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
    ]

    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )


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
    save_path = os.path.join(save_dir, f"{model_name}_best.keras")

    print(f"\n{'='*55}")
    print(f"  Phase 1 — head only  ({phase1_epochs} epochs, lr={phase1_lr})")
    print(f"{'='*55}")
    h1 = _phase1(model, train_ds, val_ds, phase1_epochs, phase1_lr, set_trainable_fn)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — full fine-tuning  (max {phase2_epochs} epochs, lr={phase2_lr})")
    print(f"  Early stopping: patience={patience} su val_srcc")
    print(f"{'='*55}")
    h2 = _phase2(model, train_ds, val_ds, phase2_epochs, phase2_lr, patience,
                 save_path, set_trainable_fn)

    print(f"\nModello migliore salvato in: {save_path}")
    return h1, h2


# ---------------------------------------------------------------------------
# Entry point rapido per test / lancio diretto
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train_ds, val_ds, test_ds = prepare_datasets(save_csv=True)

    model_a = build_model_a()
    model_a.summary()

    for images, mos in train_ds.take(1):
        preds = model_a(images, training=False)
        print("\nSmoke test pipeline + Model A:")
        print("Images shape:", images.shape)
        print("Images range:", float(tf.reduce_min(images)), float(tf.reduce_max(images)))
        print("MOS shape:", mos.shape)
        print("Predictions shape:", preds.shape)

    # train(
    #     model=model_a,
    #     train_ds=train_ds,
    #     val_ds=val_ds,
    #     model_name='model_a',
    #     set_trainable_fn=set_trainable,
    # )
