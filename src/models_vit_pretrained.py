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
    from transformers.models.swin.modeling_tf_swin import TFSwinModel
except ImportError:  # fallback for environments where TFSwinModel is not importable
    TFSwinModel = None
    from transformers.models.auto.modeling_tf_auto import TFAutoModel

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


def _apply_local_augmentation(images: tf.Tensor) -> tf.Tensor:
    """
    Local augmentation for Swin using tf.image ops only.

    This keeps the tensor shape explicit and stable, which is safer than
    Keras preprocessing layers for HF TF backbones running through tf.function.
    """
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=12.0)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    images = tf.image.random_saturation(images, lower=0.9, upper=1.1)

    # Mild crop-resize jitter to simulate zoom/translation while preserving
    # the fixed 224x224 spatial contract expected by the Swin backbone.
    batch_size = tf.shape(images)[0]
    crop_scale = tf.random.uniform((batch_size,), minval=0.88, maxval=1.0)
    crop_size = tf.cast(crop_scale * tf.cast(tf.shape(images)[1], tf.float32), tf.int32)
    crop_size = tf.maximum(crop_size, 1)

    def _crop_single(args):
        image, side = args
        cropped = tf.image.random_crop(image, size=(side, side, 3))
        return tf.image.resize(cropped, (tf.shape(images)[1], tf.shape(images)[2]))

    images = tf.map_fn(
        _crop_single,
        (images, crop_size),
        fn_output_signature=tf.float32,
    )
    return tf.clip_by_value(images, 0.0, 255.0)


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
        run_eagerly: bool = False,
        **kwargs,
    ):
        super().__init__(name="ModelC_swin_tiny_regressor", **kwargs)
        self.backbone_name = backbone_name
        self.dropout_rate = dropout
        self.use_model_augmentation = use_model_augmentation
        # Eager mode necessario per il training: il backbone TFSwin
        # (transformers 4.44) produce un errore di reshape in window_reverse
        # se il forward con training=True viene tracciato sotto tf.function.
        self.run_eagerly_training = run_eagerly

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
        image_size = self.swin_backbone.config.image_size
        if isinstance(image_size, (tuple, list)):
            self.image_size = int(image_size[0])
        else:
            self.image_size = int(image_size)

        self.regression_head = keras.Sequential(
            [
                layers.Dense(256, activation="gelu",
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.Dropout(dropout),
                layers.Dense(64, activation="gelu",
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.Dropout(dropout / 2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="swin_regression_head",
        )

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)

        if self.use_model_augmentation and training:
            x = _apply_local_augmentation(x)

        x = tf.image.resize(x, (self.image_size, self.image_size))
        x = tf.ensure_shape(x, (None, self.image_size, self.image_size, 3))

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
                "run_eagerly": self.run_eagerly_training,
            }
        )
        return config


def set_trainable_vit(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    """
    Freeze or unfreeze the Hugging Face Swin backbone.

    n_top_layers: if set, freezes all encoder stages except the last n_top_layers.
    Swin-Tiny has 4 stages [2, 2, 6, 2 blocks]; n_top_layers=2 unfreezes stage_2+stage_3.
    Pass None to unfreeze the entire backbone (Phase 2b).

    Keras ignores trainable=True on a child whose ancestor has trainable=False,
    so partial unfreezing must keep the backbone trainable and freeze the
    lower children individually — never the other way around.
    """
    if hasattr(model, "swin_backbone"):
        backbone = model.swin_backbone
    else:
        backbone = model.get_layer("swin_backbone")

    backbone.trainable = backbone_trainable

    if backbone_trainable and n_top_layers is not None:
        try:
            # TFSwinModel wraps layers in .swin (TFSwinMainLayer); fall back to direct access
            swin_main = backbone.swin if hasattr(backbone, "swin") else backbone
            swin_main.embeddings.trainable = False
            encoder_stages = swin_main.encoder.layers
            n_stages = len(encoder_stages)
            n_freeze = max(0, n_stages - n_top_layers)
            for stage in encoder_stages[:n_freeze]:
                stage.trainable = False
        except AttributeError:
            backbone.trainable = True  # fallback: full unfreeze


def build_model_vit(
    input_shape: tuple = (224, 224, 3),
    backbone_name: str = DEFAULT_BACKBONE,
    dropout: float = 0.3,
    use_model_augmentation: bool = True,
    run_eagerly: bool = False,
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
        run_eagerly=run_eagerly,
    )
