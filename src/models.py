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

def _regression_head(x: tf.Tensor, dropout: float = 0.3) -> tf.Tensor:
    """Dropout → Dense(1). Returns a scalar tensor per sample."""
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, name="mos_output")(x)
    return x


def set_trainable(model: keras.Model, backbone_trainable: bool) -> None:
    """
    Toggle the backbone trainability without recompiling the model structure.
    Call model.compile() again after this to apply the new learning rates.
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = backbone_trainable


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
        input_tensor=inputs,
    )
    backbone.trainable = False   # start frozen — Phase 1

    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)   # (B, 1280)
    outputs = _regression_head(x, dropout=dropout)                    # (B, 1)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelA_baseline")


# ---------------------------------------------------------------------------
# Model B — Multi-Scale CNN
# ---------------------------------------------------------------------------

def build_model_b(dropout: float = 0.3) -> keras.Model:
    """
    Three EfficientNetB0 branches at resolutions 224, 112, 56.
    Features are concatenated (3 × 1280 = 3840) before the regression head.

    Motivation: image quality depends on both local distortions (high res) and
    global composition (low res). Three scales capture both levels.

    Weights are SHARED across the three branches (single backbone, three
    different inputs resized before the forward pass).

    Input: a single image at 224×224 — the two lower resolutions are obtained
    by resizing inside the graph so the DataLoader stays simple.
    """
    inputs = keras.Input(shape=(224, 224, 3), name="image")

    # Shared backbone — instantiated once, called three times
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False   # start frozen — Phase 1

    # Branch 224 — original resolution
    x224 = backbone(inputs, training=False)
    f224 = layers.GlobalAveragePooling2D(name="gap_224")(x224)   # (B, 1280)

    # Branch 112 — downsample to 112×112 then pass through backbone
    x112 = layers.Resizing(112, 112, name="resize_112")(inputs)
    x112 = layers.Resizing(224, 224, name="upsample_112")(x112)  # backbone expects 224
    x112 = backbone(x112, training=False)
    f112 = layers.GlobalAveragePooling2D(name="gap_112")(x112)   # (B, 1280)

    # Branch 56 — downsample to 56×56 then pass through backbone
    x56 = layers.Resizing(56, 56, name="resize_56")(inputs)
    x56 = layers.Resizing(224, 224, name="upsample_56")(x56)
    x56 = backbone(x56, training=False)
    f56 = layers.GlobalAveragePooling2D(name="gap_56")(x56)      # (B, 1280)

    combined = layers.Concatenate(name="multi_scale_features")([f224, f112, f56])  # (B, 3840)
    outputs = _regression_head(combined, dropout=dropout)                           # (B, 1)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelB_multiscale")


def set_trainable_b(model: keras.Model, backbone_trainable: bool) -> None:
    """Toggle backbone for Model B (shared backbone called 'efficientnetb0')."""
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = backbone_trainable
