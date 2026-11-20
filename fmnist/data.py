"""Loading the official Fashion-MNIST split, preprocessing and label hiding."""
import gzip
import os
import urllib.request

import numpy as np

from .config import DATA_DIR, RANDOM_STATE

FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _read_images(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_split(split, data_dir=DATA_DIR):
    """Return (images uint8 [n,28,28], labels int64 [n]) for the official 'train' or 'test' split.

    Downloads the official Zalando files into `data_dir` if they are missing.
    """
    img_name, lbl_name = FILES[split]
    os.makedirs(data_dir, exist_ok=True)
    for name in (img_name, lbl_name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(FASHION_URL + name, path)
    images = _read_images(os.path.join(data_dir, img_name))
    labels = _read_labels(os.path.join(data_dir, lbl_name))
    expected = 60_000 if split == "train" else 10_000
    assert images.shape == (expected, 28, 28) and labels.shape == (expected,)
    return images, labels


def to_features(images):
    """Flatten 28x28 images to 784 features and scale pixels to [0, 1] (as in the notebook)."""
    images = np.asarray(images)
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def make_partial_labels(y_true, label_fraction, random_state=RANDOM_STATE):
    """Keep `label_fraction` of labels per class, mark the rest -1 (identical to the notebook).

    Returns (y_partial, labelled_indices).
    """
    rng = np.random.RandomState(random_state)
    y_partial = np.full_like(y_true, -1)
    labelled = []
    for cls in np.unique(y_true):
        cls_idx = np.where(y_true == cls)[0]
        rng.shuffle(cls_idx)
        n_keep = max(1, int(round(len(cls_idx) * label_fraction)))
        labelled.extend(cls_idx[:n_keep])
    labelled = np.array(sorted(labelled))
    y_partial[labelled] = y_true[labelled]
    return y_partial, labelled
