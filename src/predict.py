"""Inference helpers - used by the notebook AND the Streamlit app."""
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import preprocess_pil

_CAT_CLASS_IDS = {281, 282, 283, 284, 285}
_DOG_CLASS_IDS = set(range(151, 269)) | {275}
_ANIMAL_CLASS_IDS = _CAT_CLASS_IDS | _DOG_CLASS_IDS
_MIN_GATE_CONFIDENCE = 0.10


def _remove_empty_quantization_configs(value):
    if isinstance(value, dict):
        return {
            key: _remove_empty_quantization_configs(item)
            for key, item in value.items()
            if not (key == "quantization_config" and item is None)
        }
    if isinstance(value, list):
        return [_remove_empty_quantization_configs(item) for item in value]
    return value


def load_model(path):
    import tensorflow as tf

    model_path = Path(path)
    if model_path.suffix == ".keras":
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read("config.json"))
        compatible_config = _remove_empty_quantization_configs(config)

        if compatible_config != config:
            with tempfile.TemporaryDirectory() as temp_dir:
                compatible_path = Path(temp_dir) / model_path.name
                with zipfile.ZipFile(model_path) as source, zipfile.ZipFile(
                    compatible_path, "w"
                ) as compatible:
                    for entry in source.infolist():
                        contents = source.read(entry.filename)
                        if entry.filename == "config.json":
                            contents = json.dumps(compatible_config).encode("utf-8")
                        compatible.writestr(entry, contents)
                return tf.keras.models.load_model(
                    str(compatible_path), compile=False
                )

    return tf.keras.models.load_model(str(model_path), compile=False)


def load_animal_gate():
    """Load an ImageNet classifier to reject images outside the cat/dog classes."""
    import tensorflow as tf

    return tf.keras.applications.MobileNetV2(weights="imagenet")


def _is_cat_or_dog_image(im, gate_model):
    import tensorflow as tf

    image = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    scores = gate_model(batch, training=False).numpy()[0]
    class_id = int(np.argmax(scores))
    return (
        class_id in _ANIMAL_CLASS_IDS
        and float(scores[class_id]) >= _MIN_GATE_CONFIDENCE
    )


def predict_pil(model, im, gate_model, threshold=0.5):
    """Same preprocessing as training -> sigmoid probability of Dog -> label."""
    if not _is_cat_or_dog_image(im, gate_model):
        return {
            "label": None,
            "prob_dog": None,
            "prob_cat": None,
            "confidence": 0.0,
            "is_cat_or_dog": False,
        }

    arr = preprocess_pil(im)[None].astype("float32")
    p_dog = float(model(arr, training=False).numpy()[0][0])
    label = "Dog" if p_dog >= threshold else "Cat"
    return dict(label=label, prob_dog=p_dog, prob_cat=1.0 - p_dog,
                confidence=max(p_dog, 1.0 - p_dog), is_cat_or_dog=True)


def predict_path(model, path, gate_model, threshold=0.5):
    with Image.open(path) as im:
        im.load()
        return predict_pil(model, im, gate_model, threshold)
