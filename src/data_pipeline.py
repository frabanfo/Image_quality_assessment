import os
import pandas as pd

from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import tensorflow as tf

from scipy.io import loadmat
from sklearn.model_selection import train_test_split


IMG_SIZE = 224
BATCH_SIZE = 16
RANDOM_STATE = 42
DATASET_DIR_ENV = "CHALLENGEDB_DIR"
DEFAULT_DATASET_DIR = "ChallengeDB_release"

MOS_MIN = 0.0
MOS_MAX = 100.0


def load_local_env(env_path=".env"):
    if DATASET_DIR_ENV in os.environ or not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == DATASET_DIR_ENV:
                os.environ[DATASET_DIR_ENV] = value.strip().strip('"').strip("'")
                return


def resolve_dataset_paths(data_dir=None, image_dir=None):
    """
    Risolve i path del dataset.
    """
    load_local_env()
    dataset_dir = os.environ.get(DATASET_DIR_ENV, DEFAULT_DATASET_DIR)
    resolved_data_dir = data_dir or os.path.join(dataset_dir, "Data")
    resolved_image_dir = image_dir or os.path.join(dataset_dir, "Images")

    return resolved_data_dir, resolved_image_dir


def validate_dataset_paths(data_dir, image_dir):
    required_files = [
        "AllImages_release.mat",
        "AllMOS_release.mat",
        "AllStdDev_release.mat",
    ]
    missing_files = [
        os.path.join(data_dir, file_name)
        for file_name in required_files
        if not os.path.exists(os.path.join(data_dir, file_name))
    ]

    if missing_files or not os.path.isdir(image_dir):
        message = [
            "Dataset ChallengeDB non trovato o incompleto.",
            f"Data dir usata: {data_dir}",
            f"Images dir usata: {image_dir}",
            "",
            "Soluzioni:",
            f"- metti il dataset in {DEFAULT_DATASET_DIR}/",
            f"- oppure imposta {DATASET_DIR_ENV} al path locale del dataset",
            "- oppure passa data_dir e image_dir a prepare_datasets()",
        ]

        if missing_files:
            message.append("")
            message.append("File .mat mancanti:")
            message.extend(f"- {path}" for path in missing_files)

        if not os.path.isdir(image_dir):
            message.append("")
            message.append(f"Cartella immagini mancante: {image_dir}")

        raise FileNotFoundError("\n".join(message))


def build_image_index(image_dir):
    """
    Scansiona ricorsivamente image_dir e costruisce una mappa:
    nome_file -> path_completo.
    """

    image_index = {}
    duplicate_names = set()

    for root, _, files in os.walk(image_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if file_name in image_index:
                duplicate_names.add(file_name)
                continue

            image_index[file_name] = file_path

    if duplicate_names:
        print(
            "Attenzione: trovati nomi immagine duplicati nelle "
            f"sottocartelle. Verrà usata la prima occorrenza per: "
            f"{sorted(duplicate_names)[:5]}"
        )

    return image_dir, image_index
        

def set_img_size(size: int) -> None:
    """Imposta la risoluzione di input. Va chiamata prima di creare i dataset:
    IMG_SIZE è letto a trace-time dalle funzioni di caricamento."""
    global IMG_SIZE
    IMG_SIZE = int(size)


def _decode_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return image


def load_image(image_path, mos):
    """Immagine in [0,255] float32, MOS normalizzato in [0,1]."""
    image = _decode_image(image_path)
    mos = (tf.cast(mos, tf.float32) - MOS_MIN) / (MOS_MAX - MOS_MIN)
    return image, mos


def load_image_with_std(image_path, mos, std):
    """Come load_image ma restituisce y = [mos_norm, std_norm] shape (2,)."""
    image = _decode_image(image_path)
    mos_norm = (tf.cast(mos, tf.float32) - MOS_MIN) / (MOS_MAX - MOS_MIN)
    std_norm = tf.cast(std, tf.float32) / MOS_MAX
    return image, tf.stack([mos_norm, std_norm])


def _decode_image_native(image_path):
    """Decodifica senza resize: mantiene la risoluzione nativa per il crop."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image.set_shape([None, None, 3])
    return image


def load_image_native(image_path, mos):
    """Immagine a risoluzione nativa in [0,255], MOS in [0,1] (per crop nativi)."""
    image = _decode_image_native(image_path)
    mos = (tf.cast(mos, tf.float32) - MOS_MIN) / (MOS_MAX - MOS_MIN)
    return image, mos


def extract_native_patch(image: tf.Tensor, mos: tf.Tensor, crop_frac: float = 0.7):
    """
    Crop a una frazione del lato nativo, poi resize a IMG_SIZE. Croppando
    dall'originale a piena risoluzione (decode senza resize) si preserva il
    dettaglio ad alta frequenza che un resize precoce distruggerebbe — schema
    stile DeepBIQ.
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    ch = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
    y = tf.random.uniform((), 0, tf.maximum(h - ch, 1), dtype=tf.int32)
    x = tf.random.uniform((), 0, tf.maximum(w - cw, 1), dtype=tf.int32)
    patch = image[y : y + ch, x : x + cw, :]
    patch = tf.image.resize(patch, [IMG_SIZE, IMG_SIZE])
    return patch, mos


def extract_native_patch_exact(image: tf.Tensor, mos: tf.Tensor):
    """
    Crop esatto IMG_SIZE×IMG_SIZE dai pixel nativi — NESSUN resize, quindi
    nessuna interpolazione (stile DeepBIQ "puro": il crop è già la dimensione
    di input della rete). A differenza di extract_native_patch, che croppa una
    frazione del lato e poi riscala (introducendo un lieve upscale).

    Fallback: se l'immagine nativa è più piccola di IMG_SIZE su un lato, si
    riscala l'intera immagine (caso raro: quasi tutte le foto sono 500×500).
    """
    size = IMG_SIZE
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    def _crop():
        y = tf.random.uniform((), 0, tf.maximum(h - size, 0) + 1, dtype=tf.int32)
        x = tf.random.uniform((), 0, tf.maximum(w - size, 0) + 1, dtype=tf.int32)
        return image[y : y + size, x : x + size, :]

    def _resize():
        return tf.image.resize(image, [size, size])

    patch = tf.cond(tf.logical_and(h >= size, w >= size), _crop, _resize)
    patch = tf.ensure_shape(patch, [size, size, 3])
    return patch, mos


def augment_flip(image, mos):
    """Augmentation IQA-safe: solo flip orizzontale (non altera la qualità
    percepita, a differenza del jitter fotometrico)."""
    image = tf.image.random_flip_left_right(image)
    return image, mos


def augment(image, mos):
    """
    Augmentation leggera per IQA: solo trasformazioni che non alterano
    la qualità percepita (flip orizzontale, piccole variazioni di luminosità
    e contrasto). Evitare distorsioni geometriche forti o color jitter pesante
    che invaliderebbero il label MOS.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=10.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mos


def create_tf_dataset(
    df,
    batch_size=BATCH_SIZE,
    shuffle=False,
    return_std=False,
    use_augmentation=True,
    use_patch_sampling: bool = False,
    patches_per_image: int = 4,
    patch_crop_frac: float = 0.7,
    patch_exact: bool = False,
    flip_only: bool = False,
):
    """
    Converte un DataFrame in tf.data.Dataset.

    return_std=False  → emette (image, mos_norm)
    return_std=True   → emette (image, [mos_norm, std_norm]) — per WeightedMSELoss

    use_patch_sampling: if True (and shuffle=True, return_std=False), extract
    patches_per_image native crops per image via flat_map (decode senza resize).
      patch_exact=False → crop a frazione patch_crop_frac del lato + resize a IMG_SIZE.
      patch_exact=True  → crop esatto IMG_SIZE×IMG_SIZE, nessun resize (DeepBIQ puro).
    flip_only: se True applica solo il flip orizzontale (augmentation IQA-safe),
      invece della augment() fotometrica completa.
    """
    image_paths = df["image_path"].values
    mos_values = df["mos"].values

    if return_std:
        std_values = df["std"].values
        slices = (image_paths, mos_values, std_values)
    else:
        slices = (image_paths, mos_values)

    dataset = tf.data.Dataset.from_tensor_slices(slices)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )

    patch_mode = shuffle and use_patch_sampling and not return_std
    if patch_mode:
        # Crop nativi: si decodifica senza resize e si croppa dall'originale.
        map_fn = load_image_native
    else:
        map_fn = load_image_with_std if return_std else load_image
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if patch_mode:
        _n = patches_per_image
        _cf = patch_crop_frac
        if patch_exact:
            crop_fn = lambda i, m: extract_native_patch_exact(i, m)
        else:
            crop_fn = lambda i, m: extract_native_patch(i, m, crop_frac=_cf)
        dataset = dataset.flat_map(
            lambda img, mos: tf.data.Dataset.from_tensors((img, mos))
            .repeat(_n)
            .map(crop_fn)
        )

        # Rimescola i patch dopo l'espansione: senza questo un batch contiene
        # i patch consecutivi di poche immagini (pochi MOS distinti) → il termine
        # PLCC della loss diventa instabile (varianza ~0 → 0/0 = NaN).
        dataset = dataset.shuffle(
            buffer_size=1024, seed=RANDOM_STATE, reshuffle_each_iteration=True
        )

    if shuffle and (flip_only or use_augmentation):
        aug_fn = augment_flip if flip_only else augment
        if return_std:
            dataset = dataset.map(
                lambda img, y: (aug_fn(img, y[0])[0], y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_datasets(
    data_dir=None,
    image_dir=None,
    batch_size=BATCH_SIZE,
    save_csv=True,
    split_dir="Splits",
    return_std=False,
    use_augmentation=True,
    use_patch_sampling: bool = False,
    patches_per_image: int = 4,
    patch_crop_frac: float = 0.7,
    patch_exact: bool = False,
    flip_only: bool = False,
    img_size: int | None = None,
):
    """
    Funzione principale.

    Restituisce:
        train_ds, val_ds, test_ds

    Inoltre, se save_csv=True, salva anche:
        Splits/train.csv
        Splits/val.csv
        Splits/test.csv
    """
    if img_size is not None:
        set_img_size(img_size)

    # =========================
    # 1. Caricamento file .mat
    # =========================

    data_dir, image_dir = resolve_dataset_paths(data_dir, image_dir)
    validate_dataset_paths(data_dir, image_dir)

    image_dir, image_index = build_image_index(image_dir)

    images_mat = loadmat(os.path.join(data_dir, "AllImages_release.mat"))
    mos_mat = loadmat(os.path.join(data_dir, "AllMOS_release.mat"))
    std_mat = loadmat(os.path.join(data_dir, "AllStdDev_release.mat"))

    images = images_mat["AllImages_release"]
    mos = mos_mat["AllMOS_release"]
    std = std_mat["AllStdDev_release"]

    # =========================
    # 2. Estrazione nomi immagini
    # =========================

    image_names = []

    for item in images:
        name = item[0][0]
        image_names.append(str(name))

    mos_values = mos.squeeze()
    std_values = std.squeeze()

    df = pd.DataFrame({
        "image_name": image_names,
        "mos": mos_values,
        "std": std_values
    })

    image_index_df = pd.DataFrame(
        {
            "image_name": list(image_index.keys()),
            "image_path": list(image_index.values())
        }
    )

    # Associa a ogni immagine il suo path originale nel dataset
    df = df.merge(image_index_df, on="image_name", how="left")

    missing_mask = df["image_path"].isna()
    df.loc[missing_mask, "image_path"] = df.loc[missing_mask, "image_name"].apply(
        lambda name: os.path.join(image_dir, name)
    )

    # =========================
    # 3. Controllo immagini mancanti
    # =========================

    df["exists"] = df["image_path"].apply(os.path.exists)

    missing = df[df["exists"] == False]

    if len(missing) > 0:
        print("Attenzione: alcune immagini non sono state trovate:")
        print(missing[["image_name", "image_path"]].head())
    else:
        print("Tutte le immagini sono state trovate.")

    df = df.drop(columns=["exists"])

    # =========================
    # 4. Creazione classi low / medium / high
    # =========================

    p25 = df["mos"].quantile(0.25)
    p75 = df["mos"].quantile(0.75)

    print("\nPercentili MOS:")
    print("25%:", p25)
    print("75%:", p75)

    def assign_mos_class(mos_value):
        if mos_value <= p25:
            return "low"
        elif mos_value >= p75:
            return "high"
        else:
            return "medium"

    df["mos_class"] = df["mos"].apply(assign_mos_class)

    print("\nDistribuzione classi completa:")
    print(df["mos_class"].value_counts())
    print(df["mos_class"].value_counts(normalize=True))

    # =========================
    # 5. Split stratificato
    # =========================

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df["mos_class"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["mos_class"]
    )

    print("\nSplit completato:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\nDistribuzione classi nei tre split:")

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:
        print(f"\n{name}")
        print(split_df["mos_class"].value_counts())
        print(split_df["mos_class"].value_counts(normalize=True))

    # =========================
    # 6. Salvataggio CSV opzionale
    # =========================

    if save_csv:
        os.makedirs(split_dir, exist_ok=True)

        train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

        print(f"\nCSV salvati in {split_dir}/")

    # =========================
    # 7. Creazione tf.data.Dataset
    # =========================

    train_ds = create_tf_dataset(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        return_std=return_std,
        use_augmentation=use_augmentation,
        use_patch_sampling=use_patch_sampling,
        patches_per_image=patches_per_image,
        patch_crop_frac=patch_crop_frac,
        patch_exact=patch_exact,
        flip_only=flip_only,
    )

    val_ds = create_tf_dataset(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        return_std=return_std,
    )

    test_ds = create_tf_dataset(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        return_std=return_std,
    )

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_datasets()

    for images, mos in train_ds.take(1):
        print("\nBatch di esempio:")
        print("Images shape:", images.shape)
        print("MOS shape:", mos.shape)
        print("Primi MOS:", mos[:5])
