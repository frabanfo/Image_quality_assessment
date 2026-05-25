"""
Pretrained Vision Transformer regressor based on Hugging Face ViT.

Why this file uses a subclassed Keras model instead of the Functional API:
- Hugging Face TF backbones can reject Keras symbolic tensors (`KerasTensor`)
  when called during graph construction.
- With a subclassed model, the backbone is invoked inside `call(...)` on real
  TensorFlow tensors coming from the dataset during training/inference.
- This keeps the same architecture while avoiding the KerasTensor/type mismatch
  you encountered.
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras

try:
    from transformers import TFViTModel
except ImportError:  # fallback for environments where TFViTModel is not exported
    TFViTModel = None
    from transformers import TFAutoModel

layers = keras.layers


DEFAULT_BACKBONE = "google/vit-base-patch16-224-in21k"
VIT_IMAGE_MEAN = [0.5, 0.5, 0.5]
VIT_IMAGE_STD = [0.5, 0.5, 0.5]


def _normalize_for_vit(images: tf.Tensor) -> tf.Tensor:
    """Match the standard Hugging Face ViT preprocessing."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(VIT_IMAGE_MEAN, dtype=tf.float32)
    std = tf.constant(VIT_IMAGE_STD, dtype=tf.float32)
    images = tf.transpose(images, perm=[0, 3, 1, 2])
    mean = tf.reshape(mean, [1, 3, 1, 1])
    std = tf.reshape(std, [1, 3, 1, 1])
    return (images - mean) / std


class PretrainedViTRegressor(keras.Model):
    """
    ViT backbone from Hugging Face plus a custom regression head.

    The backbone is called at runtime on actual tensors, not on symbolic
    KerasTensors created by the Functional API.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(name="ModelC_vit_base_regressor", **kwargs)
        self.backbone_name = backbone_name
        self.dropout_rate = dropout

        if TFViTModel is not None:
            self.vit_backbone = TFViTModel.from_pretrained(
                backbone_name,
                name="vit_backbone",
            )
        else:
            self.vit_backbone = TFAutoModel.from_pretrained(
                backbone_name,
                name="vit_backbone",
            )
        self.vit_backbone.trainable = False

        self.regression_head = keras.Sequential(
            [
                layers.Dense(256, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(64, activation="gelu"),
                layers.Dropout(dropout / 2),
                layers.Dense(1, activation="linear"),
            ],
            name="vit_regression_head",
        )

    def call(self, inputs, training=False):
        pixel_values = _normalize_for_vit(inputs)
        backbone_outputs = self.vit_backbone(
            pixel_values=pixel_values,
            training=training,
        ).last_hidden_state

        cls_token = backbone_outputs[:, 0]
        return self.regression_head(cls_token, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_name": self.backbone_name,
                "dropout": self.dropout_rate,
            }
        )
        return config


def set_trainable_vit(model: keras.Model, backbone_trainable: bool) -> None:
    """Freeze or unfreeze the Hugging Face ViT backbone."""
    if hasattr(model, "vit_backbone"):
        model.vit_backbone.trainable = backbone_trainable
        return

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

    `input_shape` is kept for API consistency with the other model builders.
    The subclassed model is explicitly built with this shape so `summary()`
    works immediately.
    """
    del input_shape
    model = PretrainedViTRegressor(
        backbone_name=backbone_name,
        dropout=dropout,
    )
    return model
