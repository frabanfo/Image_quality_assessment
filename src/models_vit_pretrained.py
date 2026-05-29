"""
Pretrained Swin regressor based on Hugging Face Swin-Tiny.

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
    from transformers import TFSwinModel
except ImportError:  # fallback for environments where TFSwinModel is not exported
    TFSwinModel = None
    from transformers import TFAutoModel

layers = keras.layers


DEFAULT_BACKBONE = "microsoft/swin-tiny-patch4-window7-224"
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def _normalize_for_transformer(images: tf.Tensor) -> tf.Tensor:
    """Match the standard Swin/ImageNet preprocessing."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(IMAGE_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGE_STD, dtype=tf.float32)
    images = tf.transpose(images, perm=[0, 3, 1, 2])
    mean = tf.reshape(mean, [1, 3, 1, 1])
    std = tf.reshape(std, [1, 3, 1, 1])
    return (images - mean) / std


class PretrainedSwinRegressor(keras.Model):
    """
    Swin-Tiny backbone from Hugging Face plus a custom regression head.

    The model also includes local augmentation applied only during training,
    before the backbone preprocessing.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        dropout: float = 0.3,
        use_model_augmentation: bool = True,
        **kwargs,
    ):
        super().__init__(name="ModelC_swin_tiny_regressor", **kwargs)
        self.backbone_name = backbone_name
        self.dropout_rate = dropout
        self.use_model_augmentation = use_model_augmentation

        if TFSwinModel is not None:
            self.swin_backbone = TFSwinModel.from_pretrained(
                backbone_name,
                name="swin_backbone",
            )
        else:
            self.swin_backbone = TFAutoModel.from_pretrained(
                backbone_name,
                name="swin_backbone",
            )
        self.swin_backbone.trainable = False

        # Local augmentation kept inside the model to isolate experiments from
        # the shared tf.data pipeline. Transformations stay moderate so they do
        # not completely invalidate the MOS target.
        self.model_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(0.05, 0.05),
                layers.RandomRotation(0.03),
                layers.RandomZoom(0.08),
                layers.RandomContrast(0.10),
            ],
            name="swin_model_augmentation",
        )

        self.regression_head = keras.Sequential(
            [
                layers.Dense(192, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(64, activation="gelu"),
                layers.Dropout(dropout / 2),
                layers.Dense(1, activation="linear"),
            ],
            name="swin_regression_head",
        )

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)

        if self.use_model_augmentation and training:
            x = self.model_augmentation(x, training=training)

        pixel_values = _normalize_for_transformer(x)
        backbone_outputs = self.swin_backbone(
            pixel_values=pixel_values,
            training=training,
        )

        if hasattr(backbone_outputs, "pooler_output") and backbone_outputs.pooler_output is not None:
            features = backbone_outputs.pooler_output
        else:
            features = tf.reduce_mean(backbone_outputs.last_hidden_state, axis=1)

        return self.regression_head(features, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_name": self.backbone_name,
                "dropout": self.dropout_rate,
                "use_model_augmentation": self.use_model_augmentation,
            }
        )
        return config


def set_trainable_vit(model: keras.Model, backbone_trainable: bool) -> None:
    """Freeze or unfreeze the Hugging Face Swin backbone."""
    if hasattr(model, "swin_backbone"):
        model.swin_backbone.trainable = backbone_trainable
        return

    backbone = model.get_layer("swin_backbone")
    backbone.trainable = backbone_trainable


def build_model_vit(
    input_shape: tuple = (224, 224, 3),
    backbone_name: str = DEFAULT_BACKBONE,
    dropout: float = 0.3,
    use_model_augmentation: bool = True,
) -> keras.Model:
    """
    Build a pretrained Swin-Tiny regressor with an internal augmentation stage.

    `input_shape` is kept for API consistency with the other model builders.
    """
    del input_shape
    return PretrainedSwinRegressor(
        backbone_name=backbone_name,
        dropout=dropout,
        use_model_augmentation=use_model_augmentation,
    )
