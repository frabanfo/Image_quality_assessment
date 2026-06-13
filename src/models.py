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


# Backbone disponibili per build_model_a: builder + nome del layer Keras.
# I nomi layer restano quelli di default di keras.applications, così i
# checkpoint .weights.h5 già salvati continuano a caricarsi per nome.
_BACKBONES = {
    "b0": (keras.applications.EfficientNetB0, "efficientnetb0"),
    "v2s": (keras.applications.EfficientNetV2S, "efficientnetv2-s"),
}


def _find_backbone(model: keras.Model) -> keras.Model:
    for _, layer_name in _BACKBONES.values():
        try:
            return model.get_layer(layer_name)
        except ValueError:
            continue
    raise ValueError(
        f"Nessun backbone riconosciuto nel modello: attesi {[n for _, n in _BACKBONES.values()]}"
    )


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

    Il reset esplicito dei layer figli è necessario: i flag trainable=False
    messi sui figli in Phase 2a non vengono riazzerati da
    `backbone.trainable = True`, quindi senza reset la Phase 2b
    continuerebbe ad allenare solo gli ultimi n_top_layers.
    """
    backbone = _find_backbone(model)
    backbone.trainable = backbone_trainable
    if backbone_trainable:
        for layer in backbone.layers:
            layer.trainable = True
        if n_top_layers is not None:
            for layer in backbone.layers[:-n_top_layers]:
                layer.trainable = False


def build_model_a(
    input_shape: tuple = (224, 224, 3),
    dropout: float = 0.3,
    backbone: str = "b0",
) -> keras.Model:
    """
    EfficientNet pretrained on ImageNet with a single regression head.

    backbone: "b0" (EfficientNetB0, default) or "v2s" (EfficientNetV2-S).
    Both keras.applications variants include preprocessing/rescaling inside
    the model and expect inputs in [0, 255] — same data pipeline contract.

    Fine-tuning workflow:
      Phase 1 - set_trainable(model, backbone_trainable=False), LR ~1e-3
      Phase 2 - set_trainable(model, backbone_trainable=True),  LR ~1e-5
    """
    if backbone not in _BACKBONES:
        raise ValueError(f"backbone '{backbone}' non supportato: usa {list(_BACKBONES)}")
    builder, _ = _BACKBONES[backbone]

    inputs = keras.Input(shape=input_shape, name="image")

    backbone_model = builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone_model.trainable = False

    x = backbone_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)                 # (B, 1280)
    outputs = _regression_head(x, dropout=dropout)                    # (B, 1)

    name = "ModelA_baseline" if backbone == "b0" else f"ModelA_{backbone}"
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


# Layer di estrazione multiscala per build_model_b: (low /8, mid /16, top /32)
_MULTISCALE_TAPS = {
    "b0": ("block3b_add", "block5c_add", "top_activation"),
    "v2s": ("block3d_add", "block5i_add", "top_activation"),
}


def build_model_b(
    dropout: float = 0.3,
    input_shape: tuple = (224, 224, 3),
    backbone: str = "b0",
) -> keras.Model:
    """
    EfficientNet with multi-scale feature extraction via intermediate layers.

    backbone: "b0" (default) or "v2s". Features are extracted at three depths
    of the backbone on a single forward pass, then concatenated before the
    regression head (taps per backbone in _MULTISCALE_TAPS):

      low  (/8)  : low-level texture / distortion features
      mid  (/16) : mid-level structural features
      top  (/32) : high-level semantic features

    GAP collapses spatial dims → concatenated vector.

    Replaces the original resize-upsample approach, which fed blurred images
    to a backbone trained on sharp ImageNet inputs and ran three forward
    passes per sample (3× gradient accumulation on the shared weights).
    """
    if backbone not in _BACKBONES:
        raise ValueError(f"backbone '{backbone}' non supportato: usa {list(_BACKBONES)}")
    builder, backbone_layer_name = _BACKBONES[backbone]
    tap_low, tap_mid, tap_top = _MULTISCALE_TAPS[backbone]

    inputs = keras.Input(shape=input_shape, name="image")

    # Multi-output wrapper named like the keras.applications model so
    # set_trainable() / _find_backbone() keep working.
    _base = builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone_model = keras.Model(
        inputs=_base.input,
        outputs=[
            _base.get_layer(tap_low).output,
            _base.get_layer(tap_mid).output,
            _base.get_layer(tap_top).output,
        ],
        name=backbone_layer_name,
    )
    backbone_model.trainable = False   # start frozen — Phase 1

    f_low, f_mid, f_top = backbone_model(inputs, training=False)

    gap_low = layers.GlobalAveragePooling2D(name="gap_low")(f_low)    # (B, 40)
    gap_mid = layers.GlobalAveragePooling2D(name="gap_mid")(f_mid)    # (B, 112)
    gap_top = layers.GlobalAveragePooling2D(name="gap_top")(f_top)    # (B, 1280)

    combined = layers.Concatenate(name="multi_scale_features")([gap_low, gap_mid, gap_top])
    outputs = _regression_head(combined, dropout=dropout)

    name = "ModelB_multiscale" if backbone == "b0" else f"ModelB_multiscale_{backbone}"
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def set_trainable_b(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    """Toggle backbone for Model B. Same semantics as set_trainable."""
    set_trainable(model, backbone_trainable, n_top_layers=n_top_layers)
