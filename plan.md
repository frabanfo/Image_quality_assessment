# Piano ottimizzazione Swin-Tiny per IQA — branch `swin-transformers`

## Analisi diagnostica preliminare

Prima di dettagliare le ottimizzazioni, alcune osservazioni critiche emerse dall'analisi del codice:

**Gap Swin vs CNN: cause strutturali**

La CNN (EfficientNetB0) ha vantaggi sistematici rispetto alla configurazione Swin attuale:
1. Output sigmoid → predizioni sempre in [0,1], loss MSE ottimizzata senza rumore da valori fuori range
2. `set_trainable` CNN funziona davvero (partial unfreezing layer-by-layer), Swin sblocca tutto o niente
3. Phase 2b CNN ha max `30 - 5 = 25` epoche di early stopping, Swin con default `PHASE2_EPOCHS=10` ha solo `10 - 5 = 5` epoche
4. Head CNN: 1280 → 512 → 256 → 1 con L2, Swin: 768 → 192 → 64 → 1 senza L2 (drop ratio 4:1 vs 12:1 troppo aggressivo)
5. `run_eagerly=True` è obbligatorio per Swin → ogni step è più lento → si fanno meno epoche per vincoli di tempo

---

## Priorità 1 — Fix bug critici (obbligatori, alta priorità)

### BUG-1: `n_top_layers` ignorato in `set_trainable_vit`

**File**: `src/models_vit_pretrained.py`, righe 161–181

**Problema attuale**: `train.py` chiama `set_trainable_fn(model, backbone_trainable=True, n_top_layers=30)` in Phase 2a, ma `set_trainable_vit` esegue `del n_top_layers` → sblocca l'intero backbone di Swin dal primo giorno di Phase 2. Con solo 768 parametri nella testa e ~28M nel backbone Swin-Tiny, questo causa instabilità: i gradienti dalla testa piccola guidano troppi pesi.

**Struttura TFSwinModel (da HF source)**: Il backbone ha la seguente gerarchia:
```
TFSwinModel
  └── swin_backbone (TFSwinModel)
       ├── embeddings (TFSwinEmbeddings)
       ├── encoder (TFSwinEncoder)
       │    └── layers[0..3]  ← 4 stage, ognuno con N SwinLayer blocks
       │         └── blocks[0..N]  ← TFSwinLayer
       │              ├── attention (TFSwinAttention)
       │              └── intermediate/output
       ├── layernorm (LayerNormalization)
       └── pooler (Dense 768→768)
```

Swin-Tiny ha 4 stage con rispettivamente 2, 2, 6, 2 blocchi. I layer più "di alto livello" (semantici) sono `encoder.layers[3]` (stage 4, 2 blocchi) e `encoder.layers[2]` (stage 3, 6 blocchi).

**Implementazione corretta per `set_trainable_vit`**:

```python
def set_trainable_vit(
    model: keras.Model,
    backbone_trainable: bool,
    n_top_layers: int | None = None,
) -> None:
    if hasattr(model, "swin_backbone"):
        backbone = model.swin_backbone
    else:
        backbone = model.get_layer("swin_backbone")

    backbone.trainable = backbone_trainable

    if backbone_trainable and n_top_layers is not None:
        # Congela tutto, poi sblocca solo gli ultimi n_top_layers stage
        # Swin-Tiny ha 4 stage: [stage_0(2 block), stage_1(2), stage_2(6), stage_3(2)]
        # n_top_layers=1 → sblocca solo stage_3
        # n_top_layers=2 → sblocca stage_2 + stage_3
        for layer in backbone.layers:
            layer.trainable = False

        try:
            encoder_stages = backbone.encoder.layers
            n_stages = len(encoder_stages)
            n_freeze_stages = max(0, n_stages - n_top_layers)
            for stage in encoder_stages[n_freeze_stages:]:
                stage.trainable = True
            backbone.layernorm.trainable = True
            if hasattr(backbone, "pooler"):
                backbone.pooler.trainable = True
        except AttributeError:
            backbone.trainable = True  # fallback
```

**Nota**: il valore `n_top_layers=30` usato in `train.py` ha senso per EfficientNetB0, ma per Swin va passato `n_top_layers=2` (sblocca stage 2 e 3, ~8 blocchi su 12 totali).

**Impatto stimato**: Phase 2a più stabile → convergenza migliore in Phase 2b → +2-5 punti SRCC attesi.

---

### BUG-2: Output `linear` senza bound

**File**: `src/models_vit_pretrained.py`, riga 122

**Problema**: `Dense(1, activation="linear")` produce output in (-∞, +∞). Il dataset emette MOS normalizzato in [0,1]. Se le predizioni escono da questo range, MSE perde scala e il gradiente è sbilanciato. La CNN usa `sigmoid` e non ha questo problema.

**Soluzione**: cambiare `activation="linear"` → `activation="sigmoid"`.

Output in (0, 1), coerente con CNN. `collect_predictions` in `evaluate.py` moltiplica per 100 assumendo output [0,1] — funziona correttamente.

**Impatto**: immediato, elimina predizioni fuori scala e allinea Swin al comportamento CNN.

---

## Priorità 2 — Training schedule (alto impatto, bassa difficoltà)

### SCHED-1: Aumentare Phase 2 epochs e riequilibrare la divisione

**File**: `src/train_vit.py`

**Problema**: Con `PHASE2_EPOCHS=10`:
- `warmup_epochs = min(max(5, 10//4), 10) = 5`
- `remaining_epochs = 10 - 5 = 5`

Phase 2b ha solo 5 epoche. Con `PATIENCE=5`, EarlyStopping è praticamente inutile.

**Valori raccomandati**:
```python
PHASE1_EPOCHS = 5       # era 3
PHASE2_EPOCHS = 30      # era 10
PHASE1_LR     = 1e-3    # era 3e-4
PHASE2_LR     = 5e-6    # era 1e-5
PATIENCE      = 7       # era 5
```

Con `PHASE2_EPOCHS=30`:
- `warmup_epochs = 7`
- `remaining_epochs = 23` (con patience=7 → early stop realistico)

In `train.py`, cambiare la chiamata Phase 2a da `n_top_layers=30` a `n_top_layers=2` quando si usa Swin.

---

## Priorità 3 — Patch-based training (impatto massimo, complessità media)

### PATCH-1: Strategia patch durante training

**Motivazione**: Il patch sampling è l'augmentation più sicura per IQA perché:
- Una patch eredita il MOS dell'intera immagine (valido per distorsioni uniformi — rumore, blur, jpeg — la maggior parte di CLIVE)
- Moltiplica il dataset: N patch × 813 immagini → N×813 campioni per epoch
- Swin opera già a livello di patch internamente (patch4-window7)

**Implementazione in `data_pipeline.py`**:

```python
def extract_random_patch(image, mos, patch_size=192):
    """Estrae una patch random dall'immagine, mantiene il MOS."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    max_y = tf.maximum(h - patch_size, 0)
    max_x = tf.maximum(w - patch_size, 0)
    y = tf.random.uniform((), 0, tf.maximum(max_y, 1), dtype=tf.int32)
    x = tf.random.uniform((), 0, tf.maximum(max_x, 1), dtype=tf.int32)
    patch = image[y:y+patch_size, x:x+patch_size, :]
    patch = tf.image.resize(patch, [224, 224])
    return patch, mos
```

Aggiungere `use_patch_sampling: bool = False` e `patches_per_image: int = 4` a `create_tf_dataset()`:

```python
if shuffle and use_patch_sampling:
    dataset = dataset.flat_map(
        lambda img, mos: tf.data.Dataset.from_tensors((img, mos)).map(
            lambda i, m: extract_random_patch(i, m, patch_size=192)
        ).repeat(patches_per_image)
    )
```

**Inference con aggregazione patch** in `train_vit.py`:
```python
def predict_with_patch_ensemble(model, images, n_patches=8, patch_size=192):
    preds = []
    for _ in range(n_patches):
        patches = extract_random_patches_batch(images, patch_size)
        preds.append(model(patches, training=False))
    return tf.reduce_mean(tf.stack(preds), axis=0)
```

**Parametri suggeriti**:
- `patch_size=192` durante training (85% dell'immagine)
- `n_patches_per_image=4` → 813×4 = 3252 campioni per epoch
- `patch_size=224` durante inference (immagine intera) o ensemble di patch

**Caveat CLIVE**: distorsioni non uniformi (es. blur localizzato) introducono rumore nel label per patch. Accettabile — il modello impara a ragionare su porzioni locali e media all'inference.

**Impatto atteso**: +5-10 punti SRCC, riduce drasticamente l'overfitting su 813 immagini.

---

## Priorità 4 — Regression head (impatto medio, bassa difficoltà)

### HEAD-1: Ridimensionare la head e aggiungere L2

**File**: `src/models_vit_pretrained.py`

**Problema attuale**: `768 → 192 → 64 → 1` è aggressivo (ratio 4:1 per step). CNN: `1280 → 512 → 256 → 1` (ratio 2.5:1). Nessuna L2 nel Swin head.

**Head raccomandata**:
```python
self.regression_head = keras.Sequential([
    layers.Dense(256, activation="gelu",
                 kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(dropout),
    layers.Dense(64, activation="gelu",
                 kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(dropout / 2),
    layers.Dense(1, activation="sigmoid"),
], name="swin_regression_head")
```

Rapporto `768 → 256 → 64 → 1`: più graduale, L2 aiuta con dataset piccolo.

---

## Priorità 5 — Augmentation

### AUG-1: Allineare augmentation e disabilitarla con patch sampling

**Stato attuale**: `train_vit.py` chiama `prepare_datasets(use_augmentation=False)` correttamente — solo l'augmentation model-interna è attiva. Questo va mantenuto.

**Se patch sampling attivo**: disabilitare anche l'augmentation model-interna (`--disable-model-augmentation`), perché il patch sampling fornisce già sufficiente variabilità.

**AUG safe per IQA**:
- Flip orizzontale: OK
- Brightness ±5-10 su scala 0-255: OK
- Contrast 0.95-1.05: OK
- Crop-resize 0.88-1.0: OK

**AUG pericolose per IQA** (non usare):
- Rotazione / shear
- Color jitter forte (saturation, hue)
- Blur aggiuntivo (confonde il modello su un task che deve rilevare blur)
- JPEG re-compression

---

## Priorità 6 — Evaluate pipeline completa

### EVAL-1: Estendere `evaluate.py` per supportare `--model swin`

**File**: `src/evaluate.py`

**Modifiche**:

```python
# In build_or_load_model():
elif model_name in ("swin", "vit"):
    from src.models_vit_pretrained import build_model_vit
    model = build_model_vit()
    model(tf.zeros((1, 224, 224, 3)), training=False)

# In parse_args():
parser.add_argument("--model", choices=["a", "b", "swin"], default="a")

# Grad-CAM non applicabile per architetture attention-based:
# In main():
gradcam = not args.no_gradcam and args.model != "swin"
```

Il rescaling `y_true_n * 100` funziona già correttamente dopo il fix BUG-2 (output sigmoid [0,1]).

---

## Piano di implementazione: sequenza e dipendenze

```
Step 1 (1-2 ore):
  BUG-2 (output sigmoid)         → models_vit_pretrained.py, 1 riga
  SCHED-1 (aumenta epochs/LR)    → train_vit.py, 5 righe
  HEAD-1 (ridimensiona head+L2)  → models_vit_pretrained.py, ~10 righe
  → Esegui training → baseline con fix minimi

Step 2 (2-3 ore):
  BUG-1 (n_top_layers per Swin)  → models_vit_pretrained.py, ~20 righe
  → Richiede verifica a runtime: stampa backbone.encoder.layers nel smoke_test

Step 3 (4-6 ore):
  PATCH-1 (patch sampling)       → data_pipeline.py (~30 righe) + train_vit.py (~20 righe)
  → Modifica con più impatto atteso

Step 4 (1-2 ore):
  EVAL-1 (evaluate.py per Swin)  → evaluate.py (~30 righe)
  → Non blocca il training, utile per analisi post-training
```

---

## Configurazione di lancio raccomandata dopo i fix

```bash
# Step 1-2 (senza patch sampling):
python -m src.train_vit \
  --phase1-epochs 5 \
  --phase2-epochs 30 \
  --phase1-lr 1e-3 \
  --phase2-lr 5e-6 \
  --patience 7 \
  --batch-size 8 \
  --model-name model_swin_tiny_v2

# Step 3 (con patch sampling):
python -m src.train_vit \
  --phase1-epochs 5 \
  --phase2-epochs 30 \
  --phase1-lr 1e-3 \
  --phase2-lr 5e-6 \
  --patience 7 \
  --batch-size 8 \
  --patches-per-image 4 \
  --patch-size 192 \
  --disable-model-augmentation \
  --model-name model_swin_tiny_v3
```

---

## Riepilogo impatti stimati

| Ottimizzazione | File | Impatto SRCC | Rischio | Difficoltà |
|---|---|---|---|---|
| BUG-2: sigmoid output | `models_vit_pretrained.py` | +1-3 pt | Basso | Triviale (1 riga) |
| SCHED-1: più epoche | `train_vit.py` | +2-5 pt | Basso | Triviale |
| HEAD-1: head+L2 | `models_vit_pretrained.py` | +1-2 pt | Basso | Bassa |
| BUG-1: n_top_layers | `models_vit_pretrained.py` | +2-4 pt | Medio (API HF) | Media |
| PATCH-1: patch sampling | `data_pipeline.py` + `train_vit.py` | +5-10 pt | Medio (label noise) | Media |
| EVAL-1: evaluate Swin | `evaluate.py` | 0 (diagnostica) | Basso | Bassa |

**Stima realistica**:
- Step 1 alone (BUG-2 + SCHED-1 + HEAD-1): SRCC ~0.68-0.72 (da 0.64)
- Step 1+2+3 (tutti i fix): SRCC ~0.75-0.80

Gap residuo verso CNN (~0.88): strutturale, dovuto alla dimensione del dataset (~813 train) e all'inductive bias CNN vs Transformer.
