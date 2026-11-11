"""
API tests. They run on every push in CI, so they must be fast and deterministic.

Run from the project root:   pytest -v

Deterministic because: the model is called with training=False (no random
augmentation, no dropout) and the background fill uses a fixed noise seed, so
the same image gives the same probabilities every run.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main
from api.inference import Classifier

FIXTURES = Path(__file__).parent / "fixtures"

# One image per class, taken from the held-out TEST split of the training
# project - the model never saw these during training, so a correct answer
# here means something.
EXPECTED = {
    "rust.jpg": "Corn_(maize)___healthy",
    "blight.jpg": "Corn_(maize)___Northern_Leaf_Blight",
    "healthy.jpg": "Corn_(maize)___healthy",
}


@pytest.fixture(scope="module")
def client():
    """
    One TestClient for the whole file.

    WHY `with`: entering the context runs the app's lifespan (the model load).
    WHY scope="module": loading TensorFlow + the model takes seconds; doing it
    once instead of once per test keeps the suite fast.
    """
    with TestClient(main.app) as c:
        yield c


def post_image(client, path: Path, content_type: str = "image/jpeg"):
    with open(path, "rb") as f:
        return client.post("/predict", files={"file": (path.name, f, content_type)})


# ---------------------------------------------------------------- metadata
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "corn-leaf-disease-classifier"


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True, "error": None}


def test_model_info_has_expected_keys(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in ["model_version", "classes", "test_accuracy", "per_class", "training_date", "sha256"]:
        assert key in body, f"missing {key}"
    assert body["classes"] == list(EXPECTED.values())
    assert body["test_accuracy"] > 0.9


# ---------------------------------------------------------------- predictions
@pytest.mark.parametrize("filename,expected_class", EXPECTED.items())
def test_predict_returns_correct_top_class(client, filename, expected_class):
    """The real assertion: the model must get these known images right."""
    r = post_image(client, FIXTURES / filename)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == expected_class
    # The top class must also be the first (highest) entry in probabilities.
    assert next(iter(body["probabilities"])) == expected_class


@pytest.mark.parametrize("filename", EXPECTED)
def test_probabilities_sum_to_one(client, filename):
    probs = post_image(client, FIXTURES / filename).json()["probabilities"]
    assert len(probs) == 3
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_handles_rgba_png(client, tmp_path):
    """A PNG with an alpha channel has 4 channels; preprocessing must convert it to RGB."""
    from PIL import Image

    png = tmp_path / "leaf_rgba.png"
    Image.open(FIXTURES / "rust.jpg").convert("RGBA").save(png)
    r = post_image(client, png, "image/png")
    assert r.status_code == 200
    assert r.json()["predicted_class"] == EXPECTED["rust.jpg"]


# ---------------------------------------------------------------- bad input
def test_predict_rejects_text_file(client):
    # Even with a lying content type, the bytes aren't an image -> 400.
    r = client.post("/predict", files={"file": ("notes.txt", b"definitely not an image", "image/jpeg")})
    assert r.status_code == 400
    assert "not a valid image" in r.json()["detail"]


def test_predict_without_file_is_422(client):
    # 422 = FastAPI's automatic "request didn't match the declared parameters".
    r = client.post("/predict")
    assert r.status_code == 422


def test_predict_rejects_oversized_file(client):
    big = b"\xff\xd8" + b"0" * (10 * 1024 * 1024 + 10)  # just over 10 MB
    r = client.post("/predict", files={"file": ("huge.jpg", big, "image/jpeg")})
    assert r.status_code == 413


# ---------------------------------------------------------------- model failed to load
def test_unloaded_model_gives_503_not_crash(tmp_path, monkeypatch):
    """
    Point the app at an empty folder so loading fails. The app must still
    start, /health must say unhealthy with a 503, and /predict must say 503.
    """
    # Remember the working classifier so other tests aren't affected afterwards.
    working = getattr(main.app.state, "classifier", None)
    monkeypatch.setattr(main, "Classifier", lambda: Classifier(model_dir=tmp_path))
    try:
        with TestClient(main.app) as c:
            h = c.get("/health")
            assert h.status_code == 503
            assert h.json()["model_loaded"] is False
            assert "FileNotFoundError" in h.json()["error"]

            r = post_image(c, FIXTURES / "rust.jpg")
            assert r.status_code == 503
    finally:
        main.app.state.classifier = working
