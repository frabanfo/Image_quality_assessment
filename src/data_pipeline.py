import os
import shutil
import pandas as pd
import tensorflow as tf

from scipy.io import loadmat
from sklearn.model_selection import train_test_split


IMG_SIZE = 224
BATCH_SIZE = 16
RANDOM_STATE = 42


def normalize_image_dir(image_dir):
    """
    Restituisce una cartella immagini esistente, tollerando differenze
    di maiuscole/minuscole come Images/images.
    """

    if os.path.isdir(image_dir):
        return image_dir

    parent_dir = os.path.dirname(image_dir) or "."
    target_name = os.path.basename(image_dir)

    if not os.path.isdir(parent_dir):
        return image_dir

    for entry in os.listdir(parent_dir):
        if entry.lower() == target_name.lower():
            candidate = os.path.join(parent_dir, entry)
            if os.path.isdir(candidate):
                return candidate

    return image_dir


def flatten_training_images(image_dir):
    """
    Copia le immagini presenti in trainingImages dentro image_dir,
    così i path del dataset puntano sempre alla cartella principale.
    """

    image_dir = normalize_image_dir(image_dir)
    training_dir = os.path.join(image_dir, "trainingImages")

    if not os.path.isdir(training_dir):
        return image_dir

    copied_files = 0

    for file_name in os.listdir(training_dir):
        source_path = os.path.join(training_dir, file_name)
        target_path = os.path.join(image_dir, file_name)

        if not os.path.isfile(source_path):
            continue

        if os.path.exists(target_path):
            continue

        shutil.copy2(source_path, target_path)
        copied_files += 1

    if copied_files > 0:
        print(
            f"Copiate {copied_files} immagini da "
            f"{training_dir} a {image_dir}."
        )

    return image_dir


def resolve_image_path(image_dir, image_name):
    """
    Costruisce il path di un'immagine. Se non è nella root, prova anche
    nella sottocartella trainingImages.
    """

    main_path = os.path.join(image_dir, image_name)

    if os.path.exists(main_path):
        return main_path

    training_path = os.path.join(image_dir, "trainingImages", image_name)

    if os.path.exists(training_path):
        return training_path

    return main_path


def load_image(image_path, mos):
    """
    Legge un'immagine dal path, la ridimensiona e restituisce:
    immagine preprocessata, MOS
    """

    image = tf.io.read_file(image_path)

    # Se le immagini sono JPG/JPEG va bene decode_jpeg.
    # Se il dataset contiene anche PNG, BMP, ecc. usa decode_image.
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    image = tf.cast(image, tf.float32)

    # Normalizzazione semplice in [0, 1]
    image = image / 255.0

    mos = tf.cast(mos, tf.float32)

    return image, mos


def create_tf_dataset(df, batch_size=BATCH_SIZE, shuffle=False):
    """
    Converte un DataFrame in tf.data.Dataset.
    Ogni elemento sarà:
        immagine, mos
    """

    image_paths = df["image_path"].values
    mos_values = df["mos"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mos_values)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_datasets(
    data_dir="dataset/Data",
    image_dir="dataset/Images",
    batch_size=BATCH_SIZE,
    save_csv=True
):
    """
    Funzione principale.

    Restituisce:
        train_ds, val_ds, test_ds

    Inoltre, se save_csv=True, salva anche:
        dataset/Splits/train.csv
        dataset/Splits/val.csv
        dataset/Splits/test.csv
    """

    # =========================
    # 1. Caricamento file .mat
    # =========================

    image_dir = flatten_training_images(image_dir)

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

    # Percorso completo delle immagini
    df["image_path"] = df["image_name"].apply(
        lambda name: resolve_image_path(image_dir, name)
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
        split_dir = "dataset/Splits"
        os.makedirs(split_dir, exist_ok=True)

        train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

        print("\nCSV salvati in dataset/Splits/")

    # =========================
    # 7. Creazione tf.data.Dataset
    # =========================

    train_ds = create_tf_dataset(
        train_df,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = create_tf_dataset(
        val_df,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = create_tf_dataset(
        test_df,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_datasets()

    for images, mos in train_ds.take(1):
        print("\nBatch di esempio:")
        print("Images shape:", images.shape)
        print("MOS shape:", mos.shape)
        print("Primi MOS:", mos[:5])
