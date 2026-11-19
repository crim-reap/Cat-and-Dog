"""Image preprocessing shared by training AND deployment (identical pipeline)."""
import cv2
import numpy as np
from PIL import Image

from .data_cleaning import parallel_map, to_rgb

IMG_SIZE = 128
CLASS_NAMES = ["Cat", "Dog"]          # Cat = 0, Dog = 1


def preprocess_pil(im, size=IMG_SIZE):
    """PIL image -> uint8 array (size, size, 3). Scaling to [0,1] happens inside the Keras model."""
    im = to_rgb(im)
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_image(path, size=IMG_SIZE):
    with Image.open(path) as im:
        im.load()
        return preprocess_pil(im, size)


def load_dataset(paths, size=IMG_SIZE, workers=8):
    arrs = parallel_map(lambda p: load_image(p, size), paths, workers, "Loading images")
    return np.stack(arrs).astype(np.uint8)


def resize_batch(X, size):
    return np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in X])


def to_gray(X):
    X = X.astype(np.float32)
    return (X[..., 0] * 0.299 + X[..., 1] * 0.587 + X[..., 2] * 0.114).astype(np.uint8)


def flatten_normalize(X):
    """(N,H,W[,C]) uint8 -> (N, H*W*C) float32 in [0,1]."""
    return X.reshape(len(X), -1).astype(np.float32) / 255.0
