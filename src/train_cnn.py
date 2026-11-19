"""CNN / transfer-learning builders, augmentation and training loop (Keras)."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- models
def build_cnn(filters=(32, 64, 128), kernel=3, dropout=0.4, dense=128,
              batchnorm=False, global_pool=False, img_size=128, name="cnn"):
    """Custom CNN. Input = raw 0-255 pixels; Rescaling(1/255) is part of the model."""
    inp = keras.Input((img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inp)
    for f in filters:
        x = layers.Conv2D(f, kernel, padding="same", use_bias=not batchnorm)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x) if global_pool else layers.Flatten()(x)
    x = layers.Dense(dense, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)


def build_transfer(img_size=128, weights="imagenet", dropout=0.3, name="mobilenetv2_tl"):
    """MobileNetV2 backbone (frozen) + GAP + Dropout + sigmoid head."""
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                          include_top=False, weights=weights)
    base.trainable = False
    inp = keras.Input((img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inp)       # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)


def get_backbone(model):
    return next(l for l in model.layers if isinstance(l, keras.Model))


def unfreeze_top(model, n_layers=30):
    """Unfreeze the top n layers of the backbone (BatchNorm layers stay frozen)."""
    base = get_backbone(model)
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    return model


# ---------------------------------------------------------------- data
def augment_fn(img, label, max_rot=20.0, max_shear=0.15, zoom_range=0.15, shift=0.1):
    """Random flip + rotation + shear + zoom + shift (one affine transform). Training only."""
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    S = float(img.shape[0])
    theta = tf.random.uniform([], -max_rot, max_rot) * (np.pi / 180.0)
    shear = tf.random.uniform([], -max_shear, max_shear)
    zoom = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
    tx = tf.random.uniform([], -shift, shift) * S
    ty = tf.random.uniform([], -shift, shift) * S
    c, s = tf.cos(theta), tf.sin(theta)
    m00, m01 = c / zoom, (c * shear - s) / zoom
    m10, m11 = s / zoom, (s * shear + c) / zoom
    cx = (S - 1.0) / 2.0
    a2 = cx - m00 * cx - m01 * cx + tx
    b2 = cx - m10 * cx - m11 * cx + ty
    zero = tf.zeros([])
    transform = tf.reshape(tf.stack([m00, m01, a2, m10, m11, b2, zero, zero]), [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=transform,
        output_shape=tf.constant([int(S), int(S)]), fill_value=tf.constant(0.0),
        interpolation="BILINEAR", fill_mode="REFLECT")
    return tf.reshape(out, img.shape), label


def make_dataset(X, y, batch=64, training=False, augment=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, np.asarray(y, dtype="float32")))
    if training:
        ds = ds.shuffle(min(len(X), 4096), seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


# ---------------------------------------------------------------- training
def train_model(model, name, train_ds, val_ds, models_dir, logs_dir,
                epochs=25, lr=1e-3, patience=4):
    """Compile + fit with EarlyStopping / ModelCheckpoint / ReduceLROnPlateau.
    Returns (history DataFrame, seconds, checkpoint path of the BEST epoch)."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    ckpt = Path(models_dir) / f"{name}.keras"
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    secs = time.time() - t0
    hist = pd.DataFrame(h.history)
    hist.index.name = "epoch"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    hist.to_csv(Path(logs_dir) / f"{name}_history.csv")
    return hist, secs, str(ckpt)


def plot_history(hist, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = np.arange(1, len(hist) + 1)
    ax[0].plot(ep, hist["accuracy"], label="train"); ax[0].plot(ep, hist["val_accuracy"], label="validation")
    ax[0].set_title(f"{title} - accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["loss"], label="train"); ax[1].plot(ep, hist["val_loss"], label="validation")
    ax[1].set_title(f"{title} - loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.show()
