"""
Pretrained Vision Transformer regressor based on Hugging Face ViT.
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras
from transformers import TFViTModel

layers = keras.layers


DEFAULT_BACKBONE = "google/vit-base-patch16-224-in21k"
VIT_IMAGE_MEAN = [0.5, 0.5, 0.5]
VIT_IMAGE_STD = [0.5, 0.5, 0.5]


def _normalize_for_vit(images: tf.Tensor) -> tf.Tensor:
    """Match the standard Hugging Face ViT preprocessing."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(VIT_IMAGE_MEAN, dtype=tf.float32)
    std = tf.constant(VIT_IMAGE_STD, dtype=tf.float32)
    return (images - mean) / std


def set_trainable_vit(model: keras.Model, backbone_trainable: bool) -> None:
    """Freeze or unfreeze the Hugging Face ViT backbone."""
    backbone = model.get_layer("vit_backbone")
    backbone.trainable = backbone_trainable


def build_model_vit(
    input_shape: tuple = (224, 224, 3),
    backbone_name: str = DEFAULT_BACKBONE,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build a pretrained ViT regressor by replacing the classification head
    with a lightweight regression head.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    pixel_values = layers.Lambda(
        _normalize_for_vit,
        output_shape=input_shape,
        name="vit_preprocessing",
    )(inputs)

    backbone = TFViTModel.from_pretrained(
        backbone_name,
        name="vit_backbone",
    )
    backbone.trainable = False

    backbone_outputs = backbone(pixel_values=pixel_values).last_hidden_state
    cls_token = layers.Lambda(
        lambda output: output[:, 0],
        output_shape=(backbone.config.hidden_size,),
        name="vit_cls_token",
    )(backbone_outputs)

    x = layers.Dense(128, activation="gelu", name="vit_reg_dense_128")(cls_token)
    x = layers.Dropout(dropout, name="vit_reg_dropout_128")(x)
    x = layers.Dense(64, activation="gelu", name="vit_reg_dense_64")(x)
    x = layers.Dropout(0.2, name="vit_reg_dropout_64")(x)
    outputs = layers.Dense(1, activation="linear", name="mos_output")(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ModelC_vit_base_regressor",
    )
