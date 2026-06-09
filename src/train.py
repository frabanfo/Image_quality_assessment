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

    Registra anche 'val_loss' (MSE) se non già presente nei logs: permette
    di omettere validation_data da fit() ed evitare una seconda passata
    sulla validation, senza sovrascrivere la val_loss di chi la passa
    ancora (es. train_weighted, dove la loss è pesata).
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
        if 'val_loss' not in logs:
            errors = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
            logs['val_loss'] = float(sum(errors) / len(errors))
        print(f" — val_srcc: {logs['val_srcc']:.4f} — val_loss: {logs['val_loss']:.4f}")


# ---------------------------------------------------------------------------
# Fasi di training
# ---------------------------------------------------------------------------

def _default_optimizer(lr: float) -> keras.optimizers.Optimizer:
    return keras.optimizers.Adam(learning_rate=lr)


def make_iqa_loss(plcc_weight: float = 0.0):
    """
    MSE + plcc_weight · (1 − PLCC di batch).

    Il termine di correlazione ottimizza direttamente ciò che il progetto
    misura (correlazione, non errore assoluto). plcc_weight=0 → MSE pura.
    Il PLCC per batch è rumoroso con batch piccoli ma resta un segnale
    utile; con batch da 1 (ultimo batch parziale) il termine degenera a
    costante e il gradiente viene dal solo MSE.
    """

    def loss_fn(y_true, y_pred):
        yt = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        yp = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        loss = tf.reduce_mean(tf.square(yt - yp))
        if plcc_weight > 0.0:
            yt_c = yt - tf.reduce_mean(yt)
            yp_c = yp - tf.reduce_mean(yp)
            denom = tf.norm(yt_c) * tf.norm(yp_c) + 1e-8
            plcc = tf.reduce_sum(yt_c * yp_c) / denom
            loss += plcc_weight * (1.0 - plcc)
        return loss

    return loss_fn


def _log_trainable_params(model, phase_label: str) -> None:
    """Sanity check: il numero di pesi allenabili deve cambiare tra le fasi."""
    n_params = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"[{phase_label}] trainable params: {n_params:,}")


def _phase1(model, train_ds, val_ds, epochs, lr, set_trainable_fn, make_optimizer, loss_fn):
    """Backbone congelato — allena solo la testa di regressione."""
    set_trainable_fn(model, backbone_trainable=False)
    model.compile(
        optimizer=make_optimizer(lr),
        loss=loss_fn,
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )
    _log_trainable_params(model, "Phase 1")
    # Niente validation_data: SRCCCallback fa già una passata sulla val
    # e registra val_srcc + val_loss — evita di valutarla due volte.
    srcc_cb = SRCCCallback(val_ds)
    return model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[srcc_cb],
        verbose=1,
    )


def _phase2(model, train_ds, val_ds, epochs, lr, patience, save_path, set_trainable_fn,
            phase2a_n_top_layers: int = 30, phase2b_lr: float | None = None,
            make_optimizer=_default_optimizer, loss_fn='mse'):
    """
    Fine-tuning progressivo in due sub-fasi:

    2a — sblocca solo i top 30 layer del backbone (ultimi ~2 blocchi),
         LR invariato, niente early stopping. Adatta le feature di alto
         livello prima di toccare i layer più profondi. Il checkpoint su
         val_srcc è attivo anche qui: se 2a overfitta, il best non va perso.

    2b — sblocca l'intero backbone con LR ridotto (phase2b_lr, default LR/10),
         early stopping su val_srcc. LR ridotto per non distruggere le
         feature di basso livello. Il checkpoint riparte dal best di 2a
         (initial_value_threshold), così un 2b peggiore non lo sovrascrive.
    """
    warmup_epochs = min(max(5, epochs // 4), epochs)
    remaining_epochs = epochs - warmup_epochs
    if phase2b_lr is None:
        phase2b_lr = lr / 10

    # ---- Phase 2a: top-N layers/stages only ---------------------------------
    set_trainable_fn(model, backbone_trainable=True, n_top_layers=phase2a_n_top_layers)
    model.compile(
        optimizer=make_optimizer(lr),
        loss=loss_fn,
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )
    _log_trainable_params(model, "Phase 2a")
    h2a = model.fit(
        train_ds,
        epochs=warmup_epochs,
        callbacks=[
            SRCCCallback(val_ds),
            keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor='val_srcc',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ],
        verbose=1,
    )
    best_2a_srcc = max(h2a.history.get('val_srcc', [-float('inf')]))

    # ---- Phase 2b: full backbone, lower LR, early stopping ------------------
    set_trainable_fn(model, backbone_trainable=True)
    model.compile(
        optimizer=make_optimizer(phase2b_lr),
        loss=loss_fn,
        run_eagerly=getattr(model, "run_eagerly_training", False),
    )
    _log_trainable_params(model, "Phase 2b")

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
            initial_value_threshold=best_2a_srcc,
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
    phase2a_n_top_layers: int = 30,
    phase2b_lr: float | None = None,
    make_optimizer=_default_optimizer,
    plcc_weight: float = 0.0,
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
        phase2_lr       : learning rate fase 2a
        patience        : epoche senza miglioramento su val_srcc prima dello stop
        save_dir        : cartella dove salvare il modello migliore
        set_trainable_fn: usa set_trainable per ModelA, set_trainable_b per ModelB
        phase2b_lr      : learning rate fase 2b (default: phase2_lr / 10)
        make_optimizer  : factory lr -> optimizer (default Adam; AdamW per Swin)
        plcc_weight     : peso del termine 1−PLCC nella loss (0 = MSE pura)

    Returns:
        (history_phase1, history_phase2)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_best.weights.h5")
    loss_fn = make_iqa_loss(plcc_weight)

    print(f"\n{'='*55}")
    print(f"  Phase 1 — head only  ({phase1_epochs} epochs, lr={phase1_lr})")
    print(f"{'='*55}")
    h1 = _phase1(model, train_ds, val_ds, phase1_epochs, phase1_lr, set_trainable_fn,
                 make_optimizer, loss_fn)

    print(f"\n{'='*55}")
    print(f"  Phase 2 — full fine-tuning  (max {phase2_epochs} epochs, lr={phase2_lr})")
    print(f"  Early stopping: patience={patience} su val_srcc")
    print(f"{'='*55}")
    h2a, h2b = _phase2(model, train_ds, val_ds, phase2_epochs, phase2_lr, patience,
                       save_path, set_trainable_fn,
                       phase2a_n_top_layers=phase2a_n_top_layers,
                       phase2b_lr=phase2b_lr,
                       make_optimizer=make_optimizer,
                       loss_fn=loss_fn)

    print(f"\nModello migliore salvato in: {save_path}")
    return h1, h2a, h2b


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", "v2s"], default="a",
                        help="a=EfficientNetB0, b=B0 multi-scala, v2s=EfficientNetV2-S")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--plcc-weight", type=float, default=0.0,
                        help="Peso del termine 1−PLCC nella loss (0 = MSE pura)")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_training_target(model_name):
    if model_name == "a":
        return build_model_a(), "model_a", set_trainable
    if model_name == "v2s":
        return build_model_a(backbone="v2s"), "model_v2s", set_trainable

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
            plcc_weight=args.plcc_weight,
        )
