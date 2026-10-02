"""Pipeline dataset tf.data untuk klasifikasi kesegaran daging ayam.

Layout yang didukung:

A) Sudah dipisah:                 B) Satu folder (split otomatis):
   data/train/<kelas>/*.jpg          data/<kelas>/*.jpg
   data/val/<kelas>/*.jpg
   data/test/<kelas>/*.jpg (opsional)
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from config import IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def _from_dir(path, shuffle, seed, batch_size, **kwargs):
    return keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation="bilinear",
        # Center-crop ke rasio 1:1 lalu resize -> SAMA dengan preprocessing di C++.
        crop_to_aspect_ratio=True,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        **kwargs,
    )


def build_augmentation(seed: int = 42) -> keras.Sequential:
    # Augmentasi warna dibuat ringan: warna & kilap daging adalah fitur utama kesegaran.
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.1, fill_mode="reflect", seed=seed),
            layers.RandomZoom((-0.15, 0.15), fill_mode="reflect", seed=seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
            layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def class_counts(file_paths, class_names) -> np.ndarray:
    counts = np.zeros(len(class_names), dtype=np.int64)
    for p in file_paths:
        counts[class_names.index(Path(p).parent.name)] += 1
    return counts


def load_datasets(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                  seed: int = 42, augment: bool = True):
    """Return (train_ds, val_ds, test_ds|None, class_names, train_counts)."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")

    test_raw = None
    if (root / "train").is_dir():
        train_raw = _from_dir(root / "train", True, seed, batch_size)
        val_dir = root / "val" if (root / "val").is_dir() else root / "valid"
        val_raw = _from_dir(val_dir, False, seed, batch_size,
                            class_names=train_raw.class_names)
        if (root / "test").is_dir():
            test_raw = _from_dir(root / "test", False, seed, batch_size,
                                 class_names=train_raw.class_names)
    else:
        train_raw, val_raw = _from_dir(root, True, seed, batch_size,
                                       validation_split=val_split, subset="both")

    class_names = train_raw.class_names
    counts = class_counts(train_raw.file_paths, class_names)

    train_ds = train_raw
    if augment:
        aug = build_augmentation(seed)
        train_ds = train_ds.map(
            lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
            num_parallel_calls=AUTOTUNE,
        )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_raw.prefetch(AUTOTUNE)
    test_ds = test_raw.prefetch(AUTOTUNE) if test_raw is not None else None
    return train_ds, val_ds, test_ds, class_names, counts


def load_eval_dataset(data_dir: str, batch_size: int = 32, class_names=None):
    ds = _from_dir(data_dir, False, 0, batch_size, class_names=class_names)
    return ds.prefetch(AUTOTUNE), ds.class_names


def compute_class_weights(counts: np.ndarray) -> dict[int, float]:
    total = counts.sum()
    return {i: float(total / (len(counts) * c)) if c else 0.0 for i, c in enumerate(counts)}
