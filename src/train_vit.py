"""
Dedicated training script for the pretrained Swin-Tiny regressor.

Run from the project root:
    python3 -m src.train_vit --phase1-epochs 3 --phase2-epochs 10
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

# Il modello Swin deve girare sotto tf-keras (Keras 2): sotto Keras 3 il backbone
# HuggingFace (tf_keras.Model) non viene tracciato, quindi il fine-tuning non
# toccherebbe il backbone e i checkpoint non ne salverebbero i pesi.
# Va impostato prima di `import tensorflow`.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from scipy.stats import pearsonr, spearmanr
import tensorflow as tf
from tensorflow import keras

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
PHASE2_LR = 2e-5    # Phase 2a — fine-tuning transformer: 1e-5–3e-5 è il range tipico
PHASE2B_LR = 1e-5   # Phase 2b — full backbone (il vecchio lr/10 = 5e-7 era troppo basso)
PATIENCE = 7
SAVE_DIR = "checkpoints"


def make_swin_optimizer(lr: float) -> keras.optimizers.Optimizer:
    """AdamW con weight decay disaccoppiato + gradient clipping per il backbone Swin."""
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=1e-4,
        clipnorm=1.0,
    )
    if hasattr(optimizer, "exclude_from_weight_decay"):
        # Convenzione standard per transformer: niente decay su bias e LayerNorm
        optimizer.exclude_from_weight_decay(
            var_names=["bias", "gamma", "beta", "layernorm", "layer_norm"]
        )
    return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pretrained Swin-Tiny IQA regressor.")
    parser.add_argument("--backbone-name", default=DEFAULT_BACKBONE, help="Hugging Face model id.")
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--phase1-lr", type=float, default=PHASE1_LR)
    parser.add_argument("--phase2-lr", type=float, default=PHASE2_LR)
    parser.add_argument("--phase2b-lr", type=float, default=PHASE2B_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--plcc-weight", type=float, default=0.5,
                        help="Peso del termine 1−PLCC nella loss (0 = MSE pura)")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--run-eagerly",
        action="store_true",
        help="(default) Esegui il training in eager mode. Mantenuto per "
             "retrocompatibilità.",
    )
    parser.add_argument(
        "--graph-mode",
        action="store_true",
        help="Training sotto tf.function. ROTTO a 224/window7 (BUG-3: errore "
             "di reshape in window_reverse quando training=True è tracciato), "
             "ma FUNZIONA a 384/window12 (verificato con swin-base, giugno 12). "
             "In graph mode tf.function libera le attivazioni degli stage "
             "congelati: è l'unico modo per allenare swin-base@384 in 8GB.",
    )
    parser.add_argument(
        "--phase2b-top-stages",
        type=int,
        default=None,
        help="Se impostato, limita l'unfreezing della Phase 2b ai top N stage "
             "dell'encoder invece del backbone completo (es. 2 = stage 2-3). "
             "Necessario per swin-base@384 su 8GB: il backbone completo va in "
             "OOM anche a batch 2; gli stage 2-3 sono comunque ~96%% dei parametri.",
    )
    parser.add_argument(
        "--disable-model-augmentation",
        action="store_true",
        help="Disable the local augmentation applied inside the Swin model.",
    )
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=0,
        help="Native crops to extract per image (0 = disabled). When active, also pass --disable-model-augmentation.",
    )
    parser.add_argument(
        "--patch-exact",
        action="store_true",
        help="Crop esatto img-size×img-size dai pixel nativi, nessun resize (DeepBIQ puro).",
    )
    parser.add_argument(
        "--flip-augment",
        action="store_true",
        help="Applica solo il flip orizzontale in pipeline (augmentation IQA-safe).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Risoluzione di input della pipeline (default: 224). Deve "
             "combaciare con config.image_size del backbone, altrimenti il "
             "modello fa un resize interno che vanifica la risoluzione extra.",
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
        patch_exact=args.patch_exact,
        flip_only=args.flip_augment,
        img_size=args.img_size,
    )

    model_vit = build_model_vit(
        backbone_name=args.backbone_name,
        use_model_augmentation=not args.disable_model_augmentation,
        run_eagerly=not args.graph_mode,
    )
    model_vit(tf.zeros((1, args.img_size, args.img_size, 3), dtype=tf.float32), training=True)
    model_vit.summary()

    if args.smoke_test:
        for images, mos in train_ds.take(1):
            preds = model_vit(images, training=False)
            print("\nSmoke test Swin pipeline:")
            print("Images shape:", images.shape)
            print("MOS shape   :", mos.shape)
            print("Preds shape :", preds.shape)
        return

    if args.phase2b_top_stages is not None:
        # La Phase 2b chiama set_trainable_fn senza n_top_layers (= full unfreeze):
        # qui lo intercettiamo e lo limitiamo ai top N stage per stare in VRAM.
        def set_trainable_fn(model, backbone_trainable, n_top_layers=None):
            if backbone_trainable and n_top_layers is None:
                n_top_layers = args.phase2b_top_stages
            set_trainable_vit(model, backbone_trainable, n_top_layers=n_top_layers)
    else:
        set_trainable_fn = set_trainable_vit

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
        set_trainable_fn=set_trainable_fn,
        phase2a_n_top_layers=2,
        phase2b_lr=args.phase2b_lr,
        make_optimizer=make_swin_optimizer,
        plcc_weight=args.plcc_weight,
    )

    evaluate_on_test(model_vit, test_ds)


if __name__ == "__main__":
    main(parse_args())
