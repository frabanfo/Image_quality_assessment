"""
IQA Models — EfficientNetB0 baseline (A) and multi-scale variant (B).

Both models output a single float (predicted MOS).
Fine-tuning is managed externally via model.trainable and layer.trainable flags.

Usage:
    model_a = build_model_a()
    # Phase 1 — train only head
    set_trainable(model_a, backbone_trainable=False)
    model_a.compile(optimizer=Adam(1e-3), loss='mse')
    model_a.fit(...)
    # Phase 2 — full fine-tuning
    set_trainable(model_a, backbone_trainable=True)
    model_a.compile(optimizer=Adam(1e-5), loss='mse')
    model_a.fit(...)
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_head(x: tf.Tensor, dropout: float = 0.3, l2: float = 1e-4) -> tf.Tensor:
    """Dense(512) → Dropout → Dense(256) → Dropout → Dense(1, sigmoid).

    Output in [0, 1] — maps back to MOS scale via ×(MOS_MAX - MOS_MIN).
    Two hidden layers give the head enough capacity to project backbone
    features onto the MOS manifold without relying solely on the backbone.
    """
    reg = keras.regularizers.l2(l2)
    x = layers.Dense(512, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout / 2)(x)
    x = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="mos_output")(x)
    return x


def set_trainable(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    """
    Toggle backbone trainability.

    n_top_layers: if set, freeze all backbone layers except the last
    n_top_layers — useful for progressive unfreezing (Phase 2a).
    Pass None (default) to unfreeze the entire backbone (Phase 2b).
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = backbone_trainable
    if backbone_trainable and n_top_layers is not None:
        for layer in backbone.layers[:-n_top_layers]:
            layer.trainable = False


# ---------------------------------------------------------------------------
# Model A — CNN Baseline
# ---------------------------------------------------------------------------

def build_model_a(input_shape: tuple = (224, 224, 3), dropout: float = 0.3) -> keras.Model:
    """
    EfficientNetB0 pretrained on ImageNet with a single regression head.

    Fine-tuning workflow:
      Phase 1 — set_trainable(model, backbone_trainable=False), LR ~1e-3
      Phase 2 — set_trainable(model, backbone_trainable=True),  LR ~1e-5
    """
    inputs = keras.Input(shape=input_shape, name="image")

    # Preprocessing built into EfficientNetB0: expects pixel values in [0, 255]
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False   # start frozen — Phase 1

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)                 # (B, 1280)
    outputs = _regression_head(x, dropout=dropout)                    # (B, 1)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelA_baseline")


# ---------------------------------------------------------------------------
# Model B — Multi-Scale CNN
# ---------------------------------------------------------------------------

def build_model_b(dropout: float = 0.3) -> keras.Model:
    """
    EfficientNetB0 with multi-scale feature extraction via intermediate layers.

    Features are extracted at three depths of the backbone on a single forward
    pass, then concatenated before the regression head:

      block3b_add    — 28×28×40   : low-level texture / distortion features
      block5c_add    — 14×14×112  : mid-level structural features
      top_activation — 7×7×1280   : high-level semantic features

    GAP collapses spatial dims → concatenated vector (B, 1432).

    Replaces the original resize-upsample approach, which fed blurred images
    to a backbone trained on sharp ImageNet inputs and ran three forward
    passes per sample (3× gradient accumulation on the shared weights).
    """
    inputs = keras.Input(shape=(224, 224, 3), name="image")

    # Multi-output wrapper named "efficientnetb0" so set_trainable() works.
    _base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    backbone = keras.Model(
        inputs=_base.input,
        outputs=[
            _base.get_layer("block3b_add").output,     # 28×28×40
            _base.get_layer("block5c_add").output,     # 14×14×112
            _base.get_layer("top_activation").output,  # 7×7×1280
        ],
        name="efficientnetb0",
    )
    backbone.trainable = False   # start frozen — Phase 1

    f_low, f_mid, f_top = backbone(inputs, training=False)

    gap_low = layers.GlobalAveragePooling2D(name="gap_low")(f_low)    # (B, 40)
    gap_mid = layers.GlobalAveragePooling2D(name="gap_mid")(f_mid)    # (B, 112)
    gap_top = layers.GlobalAveragePooling2D(name="gap_top")(f_top)    # (B, 1280)

    combined = layers.Concatenate(name="multi_scale_features")([gap_low, gap_mid, gap_top])  # (B, 1432)
    outputs = _regression_head(combined, dropout=dropout)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelB_multiscale")


def set_trainable_b(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    """Toggle backbone for Model B. Same semantics as set_trainable."""
    set_trainable(model, backbone_trainable, n_top_layers=n_top_layers)
