"""
IQA CNN models based on EfficientNetB0.

Both models output a single float (predicted MOS).
Fine-tuning is managed externally via model.trainable and layer.trainable flags.
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


def _regression_head(x: tf.Tensor, dropout: float = 0.3) -> tf.Tensor:
    """Dropout -> Dense(1). Returns a scalar tensor per sample."""
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


def build_model_a(input_shape: tuple = (224, 224, 3), dropout: float = 0.3) -> keras.Model:
    """
    EfficientNetB0 pretrained on ImageNet with a single regression head.

    Fine-tuning workflow:
      Phase 1 - set_trainable(model, backbone_trainable=False), LR ~1e-3
      Phase 2 - set_trainable(model, backbone_trainable=True),  LR ~1e-5
    """
    inputs = keras.Input(shape=input_shape, name="image")

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    outputs = _regression_head(x, dropout=dropout)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelA_baseline")


def build_model_b(dropout: float = 0.3) -> keras.Model:
    """
    Three EfficientNetB0 branches at resolutions 224, 112, 56.
    Features are concatenated before the regression head.
    """
    inputs = keras.Input(shape=(224, 224, 3), name="image")

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False

    x224 = backbone(inputs, training=False)
    f224 = layers.GlobalAveragePooling2D(name="gap_224")(x224)

    x112 = layers.Resizing(112, 112, name="resize_112")(inputs)
    x112 = layers.Resizing(224, 224, name="upsample_112")(x112)
    x112 = backbone(x112, training=False)
    f112 = layers.GlobalAveragePooling2D(name="gap_112")(x112)

    x56 = layers.Resizing(56, 56, name="resize_56")(inputs)
    x56 = layers.Resizing(224, 224, name="upsample_56")(x56)
    x56 = backbone(x56, training=False)
    f56 = layers.GlobalAveragePooling2D(name="gap_56")(x56)

    combined = layers.Concatenate(name="multi_scale_features")([f224, f112, f56])
    outputs = _regression_head(combined, dropout=dropout)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelB_multiscale")


def set_trainable_b(model: keras.Model, backbone_trainable: bool) -> None:
    """Toggle backbone for Model B (shared backbone called 'efficientnetb0')."""
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = backbone_trainable
