"""
Standalone check of the Keras semantics behind the set_trainable_vit fix.

Builds a nested model shaped like TFSwinModel (outer backbone -> main layer ->
encoder with 4 stages + embeddings) and compares two partial-unfreeze styles:

  OLD approach — freeze the main layer, then re-enable child stages.
    * Keras 2 / tf-keras: trainable_weights gates on each ancestor's flag,
      so the frozen main layer hides everything -> 0 backbone params (BUG-1).
    * Keras 3: trainable propagates per-variable, so it happens to work.
  NEW approach — keep parents trainable, freeze embeddings + lower stages.
    * Works identically under both semantics.

Run this in the TRAINING environment to learn which semantics apply there:
    python scripts/verify_trainable_semantics.py
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

KERAS_MAJOR = int(keras.__version__.split(".")[0])


def n_trainable(layer):
    return sum(int(np.prod(w.shape)) for w in layer.trainable_weights)


class Stage(layers.Layer):
    def build(self, _):
        self.dense = layers.Dense(4)
        self.dense.build((None, 4))

    def call(self, x):
        return self.dense(x)


class Encoder(layers.Layer):
    def build(self, _):
        self.stage_layers = [Stage(name=f"stage_{i}") for i in range(4)]
        for s in self.stage_layers:
            s.build((None, 4))

    def call(self, x):
        for s in self.stage_layers:
            x = s(x)
        return x


class MainLayer(layers.Layer):  # analogue of TFSwinMainLayer (backbone.swin)
    def build(self, _):
        self.embeddings = layers.Dense(4, name="embeddings")
        self.embeddings.build((None, 4))
        self.encoder = Encoder(name="encoder")
        self.encoder.build((None, 4))

    def call(self, x):
        return self.encoder(self.embeddings(x))


class Backbone(keras.Model):  # analogue of TFSwinModel
    def __init__(self):
        super().__init__(name="swin_backbone")
        self.swin = MainLayer(name="swin")

    def call(self, x):
        return self.swin(x)


def fresh_backbone():
    b = Backbone()
    b(np.zeros((1, 4), dtype="float32"))
    return b


# --- OLD approach (the bug) -------------------------------------------------
b = fresh_backbone()
total = n_trainable(b)
b.trainable = True
for layer in b.layers:          # b.layers == [swin main layer]
    layer.trainable = False
for stage in b.swin.encoder.stage_layers[2:]:
    stage.trainable = True      # ignored: parent 'swin' is frozen
old_count = n_trainable(b)

# --- NEW approach (the fix) -------------------------------------------------
b2 = fresh_backbone()
b2.trainable = True
b2.swin.embeddings.trainable = False
for stage in b2.swin.encoder.stage_layers[:2]:
    stage.trainable = False
new_count = n_trainable(b2)

per_stage = n_trainable(fresh_backbone().swin.encoder.stage_layers[0])
expected_new = 2 * per_stage  # top 2 stages only

print(f"Keras version                  : {keras.__version__}")
print(f"total backbone params          : {total}")
print(f"OLD approach trainable params  : {old_count}")
print(f"NEW approach trainable params  : {new_count}   (expected {expected_new})")

assert new_count == expected_new, "new approach does not match expected count"

if KERAS_MAJOR < 3:
    assert old_count == 0, "expected Keras 2 to hide children under a frozen parent"
    print("\nKeras 2 semantics: OLD approach trained NOTHING in the backbone")
    print("(BUG-1 confirmed in this environment). NEW approach unfreezes the right stages.")
else:
    print("\nKeras 3 semantics: trainable propagates per-variable, so the OLD")
    print("approach happened to work here. NEW approach is correct under both.")
