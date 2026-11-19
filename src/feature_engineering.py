"""Hand-crafted features for the traditional ML models."""
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog

from .preprocessing import to_gray


def _hog(g):
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")


def hog_features(X_rgb):
    """HOG descriptor of the grayscale version of each (H,W,3) uint8 image."""
    G = to_gray(X_rgb)
    return np.asarray(Parallel(n_jobs=-1)(delayed(_hog)(g) for g in G), dtype=np.float32)
