"""
Model loading and prediction.

WHY a separate module: main.py is only about HTTP (routes, status codes,
validation). Everything about the model lives here. That means you could reuse
this file from a batch script or a different web framework without touching
any web code, and the tests exercise exactly the same preprocessing the API uses.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path

# WHY set this before importing TensorFlow: TF prints pages of C++ info logs
# at import time. Level "2" hides INFO and WARNING lines so real errors stand out.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# WHY an environment variable with a default: locally and in Docker the models
# folder sits next to api/, but a test (or a different deployment) can point
# somewhere else without editing code.
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
MODEL_FILE = "corn_disease_model.keras"

IMG_SIZE = 224  # MobileNetV2 was trained at 224x224; the model's input layer requires it

# ---------------------------------------------------------------------------
# Black-background fix - copied from the training project (src/data.py,
# fill_background with training=False). WHY it is here: in the training data,
# every Common rust photo had its background cut out and painted pure black.
# Training replaced that black with grey so the model couldn't learn
# "black corners = rust". The test set was scored WITH this replacement, so to
# reproduce the 97% test accuracy we must apply the identical step.
# These numbers must stay in sync with the training config.
# ---------------------------------------------------------------------------
BLACK_THRESHOLD = 25          # a pixel is "black" if every RGB channel is below this
BLACK_MIN_FRACTION = 0.05     # only treat the image as cut-out if >5% of it is black
BOARD_RGB = [135.0, 130.0, 136.0]  # the grey used at evaluation time
FILL_SEED = [42, 0]           # fixed noise seed -> same output every run (deterministic)


def _fill_black_background(x: tf.Tensor) -> tf.Tensor:
    """Replace a cut-out black background with grey, exactly as at test time."""
    dark = tf.cast(tf.reduce_max(x, axis=-1, keepdims=True) < BLACK_THRESHOLD, tf.float32)
    # Grow the mask 2px so the dark ring left by JPEG blur around the leaf is covered too.
    dilated = tf.nn.max_pool2d(dark, ksize=5, strides=1, padding="SAME")
    has_black_bg = tf.reduce_mean(dark, axis=[1, 2, 3], keepdims=True) > BLACK_MIN_FRACTION
    mask = dilated * tf.cast(has_black_bg, tf.float32)

    colour = tf.constant(BOARD_RGB, tf.float32)[None, None, None, :]
    noise = tf.random.stateless_normal(tf.shape(x), seed=FILL_SEED, stddev=12.0)
    fill = tf.clip_by_value(colour + noise, 0.0, 255.0)
    return x * (1.0 - mask) + fill * mask


def preprocess(image: Image.Image) -> tf.Tensor:
    """
    Turn an uploaded PIL image into the exact tensor the model saw at test time.

    1. convert("RGB"): a PNG with transparency is RGBA (4 channels) and a
       greyscale photo has 1. The model's input is (224, 224, 3) - anything
       else crashes it.
    2. resize to 224x224 with BILINEAR: the same method Keras'
       image_dataset_from_directory used during training. A different resize
       method gives slightly different pixels, and slightly different answers.
    3. float32 array, values still 0-255.
    4. add a batch dimension: models always take a BATCH of images, so one
       image becomes shape (1, 224, 224, 3).
    5. black-background fix (see above).

    What is deliberately NOT here: mobilenet_v2.preprocess_input.
    This model has that scaling built in as a layer (named
    "mobilenet_v2_preprocess", right after the input layer "image_0_255").
    Calling preprocess_input here as well would scale the pixels TWICE:
    0-255 -> -1..1 here, then -1..1 -> about -1.0 inside the model, so every
    image would look almost uniformly dark to the network. Nothing would crash -
    predictions would just be quietly wrong. Baking preprocessing into the model
    is the safer design for exactly this reason: the server can't forget it.
    """
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32)[None, ...]
    return _fill_black_background(tf.constant(x))


def _file_sha256(path: Path) -> str:
    """Hash of the model file. WHY: it identifies the exact weights being served."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Classifier:
    """
    Holds the loaded model plus everything we know about it.

    WHY a class instead of a global `model = load_model(...)` at import time:
    an import-time load would run the moment any file imports this module
    (including tests and tools), and if it failed the whole app would refuse to
    start. Here loading is an explicit step that main.py calls once at startup,
    and a failure is recorded in `self.error` instead of crashing.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = None
        self.class_names: list[str] = []
        self.metrics: dict = {}
        self.sha256: str | None = None
        self.error: str | None = None
        self.load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load model, class names and metrics. Never raises - records the error instead."""
        start = time.perf_counter()
        try:
            model_path = self.model_dir / MODEL_FILE

            # WHY read class names from the file: output index i means
            # class_names[i], in the order training used. A hand-typed list in
            # a different order would mislabel every prediction without any error.
            with open(self.model_dir / "class_names.json") as f:
                class_names = json.load(f)

            # compile=False: we only predict, so the optimizer/loss aren't needed.
            model = tf.keras.models.load_model(model_path, compile=False)

            # Sanity check: the number of names must match the number of outputs.
            # This catches a class_names.json from a different model.
            n_outputs = model.output_shape[-1]
            if n_outputs != len(class_names):
                raise ValueError(
                    f"Model has {n_outputs} outputs but class_names.json lists {len(class_names)} classes"
                )

            # metrics.json is nice-to-have, not required to serve predictions.
            try:
                with open(self.model_dir / "metrics.json") as f:
                    self.metrics = json.load(f)
            except FileNotFoundError:
                logger.warning("metrics.json not found - /model-info will have no metrics")

            self.model = model
            self.class_names = class_names
            self.sha256 = _file_sha256(model_path)

            # Warm-up: the first call to a TF model is slow (it builds its
            # compute graph). Doing one dummy prediction now means the first
            # real user doesn't pay that cost.
            self.predict(Image.new("RGB", (IMG_SIZE, IMG_SIZE)))

            self.error = None
            self.load_seconds = round(time.perf_counter() - start, 2)
            logger.info("Model loaded in %.2fs (sha256 %s)", self.load_seconds, self.sha256[:12])
        except Exception as e:  # noqa: BLE001 - we really do want to catch everything here
            # logger.exception also writes the full traceback to the server log.
            logger.exception("Model failed to load")
            self.model = None
            # Only the error TYPE goes to /health. The full message can contain
            # server file paths; that detail belongs in the log, not on a public endpoint.
            self.error = f"{type(e).__name__} while loading model (details in server log)"

    def predict(self, image: Image.Image) -> dict[str, float]:
        """
        Return {class_name: probability} for ALL classes.

        WHY model(x, training=False) instead of model.predict(x):
        - This model contains an augmentation block (random flips/rotations)
          and Dropout. training=False switches both off. With training=True
          the same image would give a different answer on every request.
        - model.predict() is built for big batches; for a single image it adds
          noticeable overhead per call. Calling the model directly is faster.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        probs = self.model(preprocess(image), training=False).numpy()[0]
        # strict=True: if the lengths ever differed, fail loudly instead of
        # silently dropping a class (zip's default stops at the shorter list).
        return {name: float(p) for name, p in zip(self.class_names, probs, strict=True)}
