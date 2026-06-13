# Architetture e pipeline dati — guida di studio

Documento di riferimento per capire **come funzionano i modelli e la pipeline
dati** del progetto IQA (Image Quality Assessment) su ChallengeDB. Pensato per
essere letto dall'alto verso il basso: prima il problema, poi i dati, poi le
architetture, poi il training, infine la valutazione.

> File di codice citati: `src/data_pipeline.py`, `src/models.py`,
> `src/models_vit_pretrained.py`, `src/train.py`, `src/train_vit.py`,
> `src/evaluate.py`.

---

## 0. Il problema in una frase

Dato **una sola immagine distorta** (niente riferimento "pulito"), predire il
**MOS** (Mean Opinion Score), cioè il voto medio di qualità percepita dato da
osservatori umani. È una **regressione** (un numero), non una classificazione.

- Metrica target: **SRCC** (Spearman) — conta che il modello *ordini* le
  immagini come gli umani, non che azzecchi il valore esatto.
- Sfida principale: dataset **piccolo** (~1169 immagini) → transfer learning
  obbligatorio (backbone pre-addestrati su ImageNet).

---

## 1. La pipeline dati (`src/data_pipeline.py`)

### 1.1 Il dataset: ChallengeDB (LIVE In the Wild)

- **1169 foto** da fotocamere mobili reali, con distorsioni **autentiche e
  miste** (mosso, rumore, sovra/sottoesposizione, compressione…).
- Risoluzione nativa: **quasi tutte 500×500** (verificato: 1160/1162 file;
  un paio di outlier 608×608 e 640×960).
- Per ogni immagine: **MOS ∈ [0, 100]** + **StdDev** dei giudizi (quanto gli
  osservatori erano d'accordo).
- I dati arrivano da file MATLAB: `AllImages_release.mat`, `AllMOS_release.mat`,
  `AllStdDev_release.mat` (caricati con `scipy.io.loadmat`).

### 1.2 Dai .mat al DataFrame, e lo split

1. Si leggono i `.mat` → si costruisce un indice `nome_file → path` scandendo
   ricorsivamente la cartella `Images/` (`build_image_index`).
2. Si crea un DataFrame con colonne `image_path`, `mos`, `std`.
3. **Split stratificato per qualità**, seedato (`RANDOM_STATE = 42`): le
   immagini sono divise in classi `low / medium / high` in base ai percentili
   25%/75% del MOS, e lo split mantiene le stesse proporzioni nei tre insiemi:

   ```
   Train: 818   Validation: 175   Test: 176
   (≈ 50% medium, 25% low, 25% high in ciascuno)
   ```

   Stratificare è importante con un dataset piccolo: garantisce che train/val/
   test coprano tutto lo spettro di qualità e che i numeri siano confrontabili
   tra run diverse (stesso seed = stesso split).

### 1.3 Normalizzazione del target

```
MOS_MIN = 0, MOS_MAX = 100
mos_norm = mos / 100          → target in [0, 1]
std_norm = std / 100          → usato solo dalla Weighted MSE
```

Il target è portato in **[0, 1]** perché la testa dei modelli finisce con una
**sigmoide** (output sempre in scala, niente predizioni assurde tipo MOS 130 o
−5). In valutazione si rimoltiplica per 100 per tornare alla scala MOS.

### 1.4 Risoluzione di input (`IMG_SIZE`, `set_img_size`)

`IMG_SIZE` è una variabile globale (default **224**) impostata da
`set_img_size(size)` *prima* di costruire i dataset; le funzioni di
caricamento la leggono a trace-time. Si controlla da CLI con `--img-size`
(es. 224, 384, 500). La risoluzione si è rivelata **una delle leve più forti**
sul punteggio.

### 1.5 I tre modi di caricare un'immagine

| Funzione | Cosa fa | Usata da |
|---|---|---|
| `load_image` | decode → **resize intero a IMG_SIZE** → [0,255] | training/eval standard |
| `load_image_with_std` | come sopra, ma `y = [mos_norm, std_norm]` | Weighted MSE |
| `load_image_native` | decode **senza resize** (tiene i 500×500) | patch nativi |

Nota sul contratto dei valori: le CNN (`keras.applications`) hanno il
**preprocessing dentro al modello** e vogliono input in **[0, 255]**. Lo Swin
invece vuole la normalizzazione ImageNet (vedi §2.4), applicata dentro al
modello stesso.

### 1.6 Patch nativi (la tecnica chiave) — `extract_native_patch`

L'idea (stile **DeepBIQ**): invece di dare in pasto l'intera immagine
ridimensionata, si estraggono più **ritagli a risoluzione nativa**, che
preservano il dettaglio ad alta frequenza (rumore, grana, artefatti) su cui si
gioca la qualità percepita.

Ci sono **due modalità** di crop (entrambe partono dal decode nativo):

```
                Originale 500×500  (decode senza resize — load_image_native)
                          │
     ┌────────────────────┴─────────────────────┐
     │  (A) extract_native_patch                 │  (B) extract_native_patch_exact
     │      crop_frac=0.7 → finestra 350×350     │      crop ESATTO 384×384
     │      poi tf.image.resize → 384×384        │      NESSUN resize
     ▼                                            ▼
 Patch 384×384 (lieve UPSCALE 350→384)        Patch 384×384 (pixel nativi 1:1)
     └──────────────── etichetta = MOS della madre (ereditato) ───────────────┘
```

**Modalità (B) `--patch-exact` è quella adottata come base.** Il crop è già
384×384 di pixel reali → **zero resampling, zero interpolazione** (stile DeepBIQ
"puro", §1.9). La modalità (A) resta come fallback per immagini più piccole
dell'input.

In entrambi i casi il punto è *cosa* dai in pasto alla rete:

- Resize dell'**intera** 500 → 384 = downscale → perdi alta frequenza ovunque.
- Crop nativo (350 o 384 di 500) → il dettaglio della sotto-regione resta nitido;
  con (B) addirittura senza il lieve upscale di (A).

> **Misurato**: a parità di tutto, exact-crop (B) + flip batte frac+resize (A):
> multiscala V2S @384 test SRCC **0.870** vs 0.842, RMSE 10.47 vs 12.52. Il mezzo
> upscale di (A) costava davvero qualcosa.

Con `patches_per_image = N`, ogni immagine genera **N crop diversi** (posizioni
casuali), ciascuno con lo **stesso MOS** della madre → il dataset si moltiplica
(es. 818 × 4 ≈ 3272 esempi di training). È data augmentation che insegna
**invarianza spaziale** alla qualità.

**Dettaglio cruciale — il reshuffle dopo l'espansione**: i crop di una stessa
immagine vengono prodotti in sequenza (`flat_map` + `repeat(N)`). Senza un
rimescolamento successivo, un batch conterrebbe i crop consecutivi di poche
immagini → pochissimi MOS distinti → il termine **PLCC della loss** (che è una
correlazione *interna al batch*) degenera (varianza ≈ 0 → 0/0 = NaN). Per
questo c'è uno `shuffle(buffer_size=1024)` dopo i crop.

### 1.7 Augmentation: cosa si può e cosa no

In IQA il **target è l'aspetto percepito**. Quindi qualunque augmentation che
*cambia l'aspetto* cambia la qualità reale → rende falso il MOS in etichetta.

- ✅ **Flip orizzontale**: specchiare non cambia la qualità → sicuro.
- ⚠️ **Brightness / contrast / saturation jitter**: sovra/sottoesposizione e
  contrasto piatto *sono* difetti di qualità → label noise.
- ❌ **Blur, rumore, ricompressione JPEG, rotazioni**: introducono proprio le
  distorsioni che il modello deve misurare → veleno per il label.

Due funzioni in pipeline:
- `augment()` → flip + brightness/contrast (fotometrica). **Da evitare con i
  patch**: il MOS del patch è già un'approssimazione (ereditato dalla madre) e il
  jitter fotometrico ci somma altro rumore. Misurato: patch **con** jitter+PLCC →
  val crolla a **0.665** (vs 0.741 senza).
- `augment_flip()` → **solo flip orizzontale** (`--flip-augment`). È
  l'augmentation IQA-safe: specchiare non cambia la qualità. È quella usata con i
  patch, perché aggiunge varietà spaziale senza toccare il label.

A queste si somma la varietà **gratis** data dalle posizioni casuali dei crop
(N crop diversi per immagine): di fatto già un'augmentation spaziale.

### 1.8 Il flusso `tf.data` completo (`create_tf_dataset`)

```
from_tensor_slices(paths, mos[, std])
   │  (se shuffle) shuffle(buffer = N esempi, seed=42)
   ▼
map(load_image | load_image_with_std | load_image_native)
   │  (se patch) flat_map: repeat(N) → extract_native_patch[_exact]
   │             shuffle(1024)            ← rompe i blocchi di crop
   │  (se augment) map(augment | augment_flip)
   ▼
batch(batch_size) → prefetch(AUTOTUNE)
```

`prepare_datasets(...)` orchestra tutto: risolve i path (da `.env`
`CHALLENGEDB_DIR`), fa lo split stratificato, salva i CSV in `Splits/` e
restituisce `train_ds, val_ds, test_ds`. **Solo il train** ha shuffle +
augmentation + patch; val e test sono deterministici e ordinati come i loro CSV
(serve per allineare le predizioni in valutazione).

### 1.9 Confronto con DeepBIQ (da dove viene l'idea)

DeepBIQ (Bianco et al., 2018, *"On the use of deep learning for blind image
quality assessment"*) è il riferimento da cui prendiamo l'idea dei crop nativi.
Funziona così:

1. **Crop a risoluzione nativa, esattamente la input size della rete** (224×224
   per la loro CNN) → nessun resize, nessuna interpolazione.
2. Ogni crop → CNN → predizione (o feature). Due varianti: feature extractor +
   **SVR**, oppure CNN fine-tunata end-to-end.
3. **Aggregazione a test-time**: si campionano *molte* sotto-regioni per immagine
   e si **media** la predizione → è il cuore del metodo, ciò che dà robustezza.
4. In training ogni crop **eredita il MOS** dell'immagine madre (come noi).

Cosa abbiamo preso e cosa no:

| Aspetto | DeepBIQ | Noi |
|---|---|---|
| Crop a risoluzione nativa | ✅ | ✅ |
| Crop = input size, **no resize** | ✅ (224 esatti) | ✅ con `--patch-exact` (384 esatti) |
| Crop in **training** (data aug) | ✅ | ✅ |
| **Aggregazione multi-crop a test-time** | ✅ (centrale) | ❌ provata e **scartata** |
| Feature + **SVR** | ✅ (variante) | ❌ testa end-to-end |

Due differenze meritano una nota:

- **Risoluzione**: DeepBIQ croppa a 224 (input della loro rete) su immagini
  grandi → tanti crop possibili a costo zero di resampling. Noi a 384 su immagini
  500×500 abbiamo meno margine; per questo `--patch-crop-frac` (crop + resize)
  esisteva come alternativa più flessibile, poi superata da `--patch-exact`.
- **Multi-crop a inferenza**: in DeepBIQ è il meccanismo principale; noi l'abbiamo
  testato anche sul modello exact e **non aiuta l'SRCC**. Misurato sul multiscala
  exact (test):

  | strategia inferenza | SRCC | RMSE |
  |---|---:|---:|
  | **full-resize (intera → 384)** | **0.870** | 10.47 |
  | center-crop 384 | 0.831 | 10.29 |
  | multi-crop ×8 | 0.839 | 10.24 |
  | multi-crop ×8 + full | 0.846 | 10.20 |

  La **vista intera** dà il ranking migliore: un singolo crop perde il **contesto
  globale** (composizione, esposizione complessiva) che serve a ordinare la
  qualità. I crop danno RMSE un filo più basso (più accurati sul valore assoluto)
  ma SRCC peggiore — e il target è SRCC. In DeepBIQ il guadagno del multi-crop
  veniva da crop 224 più "stretti" su una rete a bassa risoluzione; a 384 una
  vista cattura già abbastanza.

---

## 2. Le architetture (`src/models.py`, `src/models_vit_pretrained.py`)

Tre famiglie, tutte con backbone pre-addestrato su ImageNet (transfer learning)
e una testa di regressione che finisce con **sigmoide → [0,1]**.

### 2.0 La testa di regressione condivisa (CNN) — `_regression_head`

```
features → Dense(512, relu, L2) → Dropout(0.3)
         → Dense(256, relu, L2) → Dropout(0.15)
         → Dense(1, sigmoid)                     → MOS normalizzato in [0,1]
```

Due hidden layer danno alla testa abbastanza capacità per proiettare le feature
del backbone sul "manifold" del MOS senza dipendere solo dal backbone. L2 1e-4
+ dropout = regolarizzazione (dataset piccolo → rischio overfit).

### 2.1 Model A — CNN single-scale (baseline) — `build_model_a`

```
Input (H,W,3) in [0,255]
   ▼
EfficientNet  (B0  oppure  V2-S)        ← preprocessing interno
   ▼  output finale del backbone (es. 12×12×1280 a 384px per V2S)
GlobalAveragePooling2D                  → vettore (B, 1280)
   ▼
_regression_head                        → (B, 1)
```

- `backbone="b0"` → EfficientNetB0 (~4M param backbone); `backbone="v2s"` →
  EfficientNetV2-S (~20M, più capacità).
- **Single-scale**: usa **solo le feature finali** del backbone (le più
  astratte/semantiche). È il **riferimento**: backbone forte, ma una sola scala.

### 2.2 Model B — CNN multiscala (la variante "nostra") — `build_model_b`

```
Input (H,W,3) in [0,255]
   ▼
EfficientNet  (multi-output: 3 tap a profondità diverse, UN solo forward pass)
   ├─ tap basso  (/8)   → GAP → vettore "texture"      (es. 40 dim su B0)
   ├─ tap medio  (/16)  → GAP → vettore "struttura"    (es. 112 dim)
   └─ tap alto   (/32)  → GAP → vettore "semantica"    (es. 1280 dim)
   ▼
Concatenate([low, mid, top])            → vettore multi-scala
   ▼
_regression_head                        → (B, 1)
```

I layer "spillati" (`_MULTISCALE_TAPS`):

| backbone | basso (/8) | medio (/16) | alto (/32) |
|---|---|---|---|
| `b0`  | `block3b_add` | `block5c_add` | `top_activation` |
| `v2s` | `block3d_add` | `block5i_add` | `top_activation` |

**Perché funziona**: le distorsioni vivono a **scale diverse** — rumore e grana
sono fenomeni *locali* (catturati ai tap bassi), una composizione povera o un
soggetto sfocato sono *globali* (tap alti). Concatenando le tre scale, la testa
vede contemporaneamente "com'è la grana" e "cos'è il soggetto".

> Nota: questa versione fa **un solo forward pass** col backbone multi-output.
> Sostituisce un vecchio approccio (rimosso) che ridimensionava l'immagine a 3
> scale e faceva 3 forward pass — dava in pasto immagini sfocate a un backbone
> addestrato su immagini nitide, e triplicava i gradienti sugli stessi pesi.

### 2.3 `set_trainable` / `set_trainable_b` — congelamento selettivo

Gestiscono quali layer del backbone sono allenabili (vedi §3). Punto delicato
(Keras 3): rimettere `backbone.trainable = True` **non** ripristina i flag dei
layer figli messi a `False` in una fase precedente → senza un **reset esplicito
dei figli** la "fine-tuning completa" allenerebbe solo gli ultimi N layer. Il
codice resetta esplicitamente tutti i figli a `trainable=True` prima di
ricongelare.

### 2.4 Model "C" — Vision Transformer Swin — `PretrainedSwinRegressor`

```
Input (B,H,W,3) in [0,255]
   │  (solo training, se attiva) augmentation locale: flip, brightness,
   │     contrast, saturation, crop-resize jitter
   ▼
resize a image_size del backbone (224 per Swin-Tiny, 384 per Swin-Base)
   ▼
normalizzazione ImageNet:  (x/255 − mean) / std,  layout NCHW
   mean=[0.485,0.456,0.406]  std=[0.229,0.224,0.225]
   ▼
TFSwinModel (HuggingFace)  →  pooler_output   (feature globale)
   ▼
testa: Dense(256, gelu, L2) → Dropout(0.3)
       Dense(64,  gelu, L2) → Dropout(0.15)
       Dense(1, sigmoid)                        → (B,1)
```

Particolarità tecniche importanti:

- **Modello subclassed** (non Functional API): `TFSwinModel` rifiuta i
  `KerasTensor` simbolici, quindi il backbone va invocato dentro `call(...)` su
  tensori reali. Per questo lo Swin "vive" in un file a parte.
- **tf-keras (Keras 2) obbligatorio**: sotto Keras 3 il backbone HF non viene
  tracciato → i suoi pesi non finirebbero nei checkpoint. `train_vit.py` forza
  `TF_USE_LEGACY_KERAS=1`. Conseguenza: i `.weights.h5` dello Swin sono in
  formato tf-keras e **non** sono leggibili da Keras 3 (e viceversa per la CNN).
- **Swin gerarchico a 4 stadi** (`[2, 2, 6, 2]` blocchi per Swin-Tiny): attention
  su **finestre locali shiftate**, costo lineare nella risoluzione, piramide di
  feature stile CNN.
- Lo Swin **ridimensiona internamente** all'image_size del backbone: se la
  pipeline manda 384 ma il backbone è window7-224, i pixel extra vengono buttati
  → per sfruttare la risoluzione serve un backbone nativo (es.
  `swin-base-patch4-window12-384`).

**Domanda di ricerca del progetto**: l'*inductive bias* convolutivo è un
vantaggio su dataset piccolo? (Spoiler nei risultati: sì, la CNN multiscala
resta davanti.)

---

## 3. Training: fine-tuning progressivo (`src/train.py`, `src/train_vit.py`)

Stessa filosofia per CNN e Swin: scongelare il backbone *gradualmente* per non
distruggere le feature pre-addestrate con i gradienti enormi di una testa
inizializzata a caso (catastrophic forgetting).

| Fase | Cosa si allena | LR | Note |
|---|---|---|---|
| **Phase 1** | solo la testa (backbone congelato) | ~1e-3 | adatta la testa random |
| **Phase 2a** | top-N del backbone (top-30 layer CNN / stadi 2–3 Swin) | basso | warmup, niente early stopping |
| **Phase 2b** | backbone (quasi) intero | ridotto (LR/10 o `--phase2b-lr`) | early stopping su `val_srcc` |

Dettagli:

- **Ottimizzatori**: Adam per la CNN; **AdamW** (weight decay 1e-4, clipnorm
  1.0, niente decay su bias/LayerNorm) per lo Swin, con `--phase2-lr 2e-5` /
  `--phase2b-lr 1e-5`.
- **Checkpoint**: `ModelCheckpoint` su `val_srcc` salva il best. Quello di 2b
  riparte dal best di 2a (`initial_value_threshold`), così un 2b peggiore non
  sovrascrive il buono.
- **ReduceLROnPlateau** su `val_srcc` (×0.5, patience 3) in 2b.

### 3.1 La metrica di selezione: `SRCCCallback`

Keras non ha lo SRCC built-in. Un callback custom, a fine epoca, fa una passata
sulla validation, calcola lo **Spearman** e lo registra come `val_srcc` (su cui
agiscono early stopping e checkpoint). Registra anche `val_loss` da solo, così
`fit()` può essere chiamato **senza** `validation_data` ed evitare una seconda
passata. Si seleziona su SRCC e **non** su val_loss: un modello può avere MSE
basso ma ranking mediocre — e a noi interessa il ranking.

### 3.2 La funzione di loss: `make_iqa_loss(plcc_weight)`

```
loss = MSE(y, ŷ)  +  plcc_weight · (1 − PLCC_di_batch)
```

- `plcc_weight = 0` → **MSE pura** (default CNN).
- `plcc_weight > 0` → aggiunge un termine che **ottimizza direttamente la
  correlazione** (ciò che misuriamo), non solo l'errore assoluto. Default Swin:
  0.5. Il PLCC è calcolato *dentro il batch*, quindi è rumoroso con batch
  piccoli; è clippato in [−1, 1] per evitare gradienti NaN quando le predizioni
  sono quasi costanti a inizio training.

### 3.3 Variante: Weighted MSE (`src/train_weighted.py`)

Pesa ogni esempio con `w = exp(−α·std)`: gli errori su immagini con **giudizi
concordi** (std bassa) pesano di più, le label rumorose (std alta) pesano meno.
Usa la StdDev — l'incertezza misurata del label — che di solito viene ignorata.
Richiede `return_std=True` nella pipeline (`load_image_with_std`).

---

## 4. Valutazione (`src/evaluate.py`)

- **Metriche**: SRCC (target), PLCC, RMSE, MAE, MSE — globali e **stratificate**
  per fascia di MOS e per fascia di StdDev degli osservatori.
- **Grafici**: scatter predetto-vs-reale, errore-vs-StdDev, Bland-Altman, RMSE/
  SRCC per fascia.
- **Grad-CAM** (solo CNN): mostra *dove guarda* la rete → si verifica che si
  concentri sulle zone degradate (sfocate/rumorose/bruciate) e non su scorciatoie
  statistiche.
- L'eval gira a una risoluzione coerente col training (`--img-size`); per lo
  Swin serve impostare `TF_USE_LEGACY_KERAS=1` (lo fa da solo annusando `--model
  swin`).

---

## 5. Quadro dei risultati (test set, split seedato)

| Modello | Caratteristica | Ris. | Test SRCC | Test PLCC |
|---|---|---:|---:|---:|
| **Multiscala V2S + crop nativi (exact)** | multi-scala + crop nativi 384 | 384 | **0.870** | **0.887** |
| Multiscala V2S + crop nativi (frac+resize) | multi-scala | 384 | 0.842 | 0.871 |
| Multiscala V2S (no crop) | multi-scala | 384 | 0.8176 | 0.8517 |
| Ensemble Model B (×3, z-score) | multi-scala B0 | 224 | 0.8023 | 0.8485 |
| Swin-Base | Vision Transformer | 384 | 0.7638 | 0.8192 |
| **Single-scale V2S + crop nativi (exact)** | single-scala | 384 | 0.768 | 0.803 |
| Baseline single-scale V2S (no crop) | single-scala | 384 | 0.729 | 0.780 |
| Model A (B0 single-scala) | single-scala | 224 | 0.630 | — |

**Le tre leve che contano**:
1. **Multiscala > single-scale**: a parità di tutto (V2S, 384, crop nativi exact,
   flip) il multiscala dà **+0.102 SRCC** (0.768 → 0.870) — la scelta
   architetturale più importante.
2. **Risoluzione**: 224 → 384 → 500 migliora in modo ~monotono (limite =
   memoria GPU).
3. **Crop nativi** in training: il crop esatto 384×384 (no resampling) porta il
   best a **0.870**, sopra al crop frazione+resize (0.842).

> Il confronto in corso (`v2s` vs `bv2s`, **entrambi con patch nativi @384**)
> serve a isolare la sola leva *multiscala* sopra la stessa ricetta di training.

---

## 6. Glossario rapido

- **NR-IQA**: No-Reference IQA — qualità senza immagine di riferimento.
- **MOS / StdDev**: voto medio di qualità / dispersione dei giudizi umani.
- **SRCC**: Spearman — correlazione tra *ranking* (metrica target).
- **PLCC**: Pearson — correlazione lineare dei valori.
- **GAP**: Global Average Pooling — collassa una mappa H×W×C in un vettore C.
- **Tap multi-scala**: estrazione di feature da layer intermedi del backbone.
- **Patch nativo**: ritaglio preso dall'originale a piena risoluzione, poi
  ridimensionato all'input del modello.
- **Catastrophic forgetting**: il backbone pre-addestrato perde le sue feature
  se aggiornato con gradienti troppo forti → si previene col congelamento
  progressivo.
```
