"""
IQA Models — EfficientNetB0 baselines and a small ViT regressor.

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

    model_vit = build_model_vit()
    model_vit.compile(optimizer=Adam(1e-4), loss='mse')
    model_vit.fit(...)
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
MLP_DIM = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_head(x: tf.Tensor, dropout: float = 0.3) -> tf.Tensor:
    """Dropout → Dense(1). Returns a scalar tensor per sample."""
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, name="mos_output")(x)
    return x


def _mlp(x: tf.Tensor, hidden_units: list, dropout_rate: float) -> tf.Tensor:
    """Simple MLP block used inside the ViT encoder and regression head."""
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def set_trainable(model: keras.Model, backbone_trainable: bool) -> None:
    """
    Toggle the backbone trainability without recompiling the model structure.
    Call model.compile() again after this to apply the new learning rates.
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = backbone_trainable


def set_trainable_vit(model: keras.Model, backbone_trainable: bool) -> None:
    """
    No-op helper to keep the same training interface for transformer models.
    ViT has no pretrained CNN backbone to freeze/unfreeze here.
    """
    del model, backbone_trainable


def _extract_patches(x: tf.Tensor, patch_size: int = PATCH_SIZE) -> tf.Tensor:
    """Split an image tensor into non-overlapping flattened patches."""
    patch_dims = patch_size * patch_size * int(x.shape[-1])
    patches = layers.Lambda(
        lambda images: tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        ),
        name="patches",
    )(x)
    return layers.Reshape((NUM_PATCHES, patch_dims), name="patches_reshape")(patches)


def _encode_patches(
    patches: tf.Tensor,
    num_patches: int = NUM_PATCHES,
    projection_dim: int = PROJECTION_DIM,
) -> tf.Tensor:
    """Project patches to tokens and add learnable positional embeddings."""
    projected = layers.Dense(
        projection_dim,
        name="patch_projection",
    )(patches)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embeddings = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
        name="patch_position_embedding",
    )(positions)
    return layers.Add(name="patch_encoder")([projected, position_embeddings])


def _transformer_block(x: tf.Tensor, layer_idx: int) -> tf.Tensor:
    """Single pre-norm transformer encoder block."""
    attention_input = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"transformer_norm_1_{layer_idx}",
    )(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=PROJECTION_DIM,
        dropout=0.1,
        name=f"transformer_mha_{layer_idx}",
    )(attention_input, attention_input)
    x = layers.Add(name=f"transformer_skip_1_{layer_idx}")([attention_output, x])

    mlp_input = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"transformer_norm_2_{layer_idx}",
    )(x)
    mlp_output = _mlp(
        mlp_input,
        hidden_units=[MLP_DIM, PROJECTION_DIM],
        dropout_rate=0.1,
    )
    return layers.Add(name=f"transformer_skip_2_{layer_idx}")([mlp_output, x])


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


# ---------------------------------------------------------------------------
# Model C — Small ViT Regressor
# ---------------------------------------------------------------------------

def build_model_vit(
    input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3),
) -> keras.Model:
    """
    Small Vision Transformer for image quality regression.

    Input: image tensor with shape (224, 224, 3).
    Output: single scalar MOS prediction.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    patches = _extract_patches(inputs)
    x = _encode_patches(patches)

    for layer_idx in range(TRANSFORMER_LAYERS):
        x = _transformer_block(x, layer_idx)

    x = layers.LayerNormalization(epsilon=1e-6, name="vit_output_norm")(x)
    x = layers.GlobalAveragePooling1D(name="vit_gap")(x)
    x = layers.Dense(128, activation="gelu", name="vit_dense_128")(x)
    x = layers.Dropout(0.3, name="vit_dropout_128")(x)
    x = layers.Dense(64, activation="gelu", name="vit_dense_64")(x)
    x = layers.Dropout(0.2, name="vit_dropout_64")(x)
    outputs = layers.Dense(1, activation="linear", name="mos_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="ModelC_small_vit")
