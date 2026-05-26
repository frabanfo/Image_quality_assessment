"""
Pretrained DeiT regressor based on Hugging Face DeiT-tiny.

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
    from transformers import TFDeiTModel
except ImportError:  # fallback for environments where TFDeiTModel is not exported
    TFDeiTModel = None
    from transformers import TFAutoModel

layers = keras.layers


DEFAULT_BACKBONE = "facebook/deit-tiny-patch16-224"
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def _normalize_for_transformer(images: tf.Tensor) -> tf.Tensor:
    """Match the standard DeiT/ImageNet preprocessing."""
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant(IMAGE_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGE_STD, dtype=tf.float32)
    images = tf.transpose(images, perm=[0, 3, 1, 2])
    mean = tf.reshape(mean, [1, 3, 1, 1])
    std = tf.reshape(std, [1, 3, 1, 1])
    return (images - mean) / std


class PretrainedDeiTRegressor(keras.Model):
    """
    DeiT-tiny backbone from Hugging Face plus a custom regression head.

    The backbone is called at runtime on actual tensors, not on symbolic
    KerasTensors created by the Functional API.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(name="ModelC_deit_tiny_regressor", **kwargs)
        self.backbone_name = backbone_name
        self.dropout_rate = dropout

        if TFDeiTModel is not None:
            self.vit_backbone = TFDeiTModel.from_pretrained(
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
                layers.Dense(1, activation="sigmoid"),
            ],
            name="vit_regression_head",
        )

    def call(self, inputs, training=False):
        pixel_values = _normalize_for_transformer(inputs)
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


# Numero di encoder layer da scongelare in phase 2a.
# DeiT-tiny ha 12 encoder layer — scongeliamo solo gli ultimi 3.
_N_TOP_ENCODER_LAYERS = 3


def set_trainable_vit(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    """Freeze/unfreeze il backbone DeiT con progressive unfreezing.

    Phase 1  (backbone_trainable=False):
        backbone completamente congelato.

    Phase 2a (backbone_trainable=True, n_top_layers=N):
        backbone congelato, poi si scongelano solo gli ultimi
        _N_TOP_ENCODER_LAYERS layer dell'encoder. Ignora il valore
        numerico di n_top_layers (calibrato per EfficientNet) e usa
        la costante specifica per DeiT.

    Phase 2b (backbone_trainable=True, n_top_layers=None):
        backbone completamente scongelato.
    """
    backbone = (
        model.vit_backbone
        if hasattr(model, "vit_backbone")
        else model.get_layer("vit_backbone")
    )

    backbone.trainable = backbone_trainable

    if backbone_trainable and n_top_layers is not None:
        # Congela tutto il backbone, poi sblocca solo gli ultimi encoder layer
        backbone.trainable = False
        encoder_layers = backbone.deit.encoder.layer
        for layer in encoder_layers[-_N_TOP_ENCODER_LAYERS:]:
            layer.trainable = True


def build_model_vit(
    input_shape: tuple = (224, 224, 3),
    backbone_name: str = DEFAULT_BACKBONE,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build a pretrained DeiT regressor by replacing the classification head
    with a lightweight regression head.

    `input_shape` is kept for API consistency with the other model builders.
    The subclassed model is explicitly built with this shape so `summary()`
    works immediately.
    """
    del input_shape
    model = PretrainedDeiTRegressor(
        backbone_name=backbone_name,
        dropout=dropout,
    )
    return model
