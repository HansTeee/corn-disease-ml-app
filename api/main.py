"""
FastAPI app: the HTTP layer around the classifier.

Run locally (from the project root, with the venv active):
    uvicorn api.main:app --port 8000
Then open http://127.0.0.1:8000/docs
"""
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated  # FastAPI's recommended way to attach File(...) to a parameter

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.inference import IMG_SIZE, MODEL_FILE, Classifier
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
)

# WHY configure logging: without this, our logger.info/exception calls go
# nowhere useful. In Docker these lines end up in `docker logs`.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

API_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB - a phone photo is 2-5 MB

# WHY cap pixels too: a tiny, highly-compressed file can decode into a gigantic
# image (a "decompression bomb") and eat all the server's RAM. Pillow raises
# DecompressionBombError above 2x this number; 50 MP is far beyond any camera
# photo someone would upload here.
Image.MAX_IMAGE_PIXELS = 25_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs once at startup, code after it once at shutdown.

    WHY load here and not inside /predict: loading takes seconds. Per request,
    every user would wait that long and memory would balloon. Loaded once, each
    prediction only pays for the forward pass.

    WHY lifespan and not @app.on_event("startup"): on_event is deprecated in
    FastAPI. lifespan keeps startup and shutdown in one function.

    If loading fails, Classifier.load() records the error instead of raising,
    so the app still starts and /health can REPORT the problem. A crashing app
    would just restart in a loop and you'd have to dig through logs to see why.
    """
    classifier = Classifier()
    classifier.load()
    app.state.classifier = classifier  # app.state: a place to keep shared objects
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Corn Leaf Disease Classifier API",
    version=API_VERSION,
    description="Classifies a corn leaf photo as Common rust, Northern Leaf Blight or Healthy.",
    lifespan=lifespan,
)


def get_classifier(request: Request) -> Classifier:
    return request.app.state.classifier


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler for any bug we didn't anticipate.

    WHY: the full traceback goes to the server log (where we can debug it),
    while the client gets a generic message. Tracebacks reveal file paths,
    library versions and code structure - useful to an attacker, useless to a user.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """
    Refuse a too-large upload BEFORE reading the body.

    WHY a middleware: by the time /predict runs, FastAPI has already received
    and parsed the whole multipart body. Checking the Content-Length header
    first means a 2 GB upload is turned away immediately. (The endpoint checks
    the real size too, because a client can omit or lie about this header.)
    """
    if request.url.path == "/predict":
        length = request.headers.get("content-length")
        # + 64 KB of slack for the multipart boundaries and headers around the file
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
            return JSONResponse(status_code=413, content={"detail": "File too large (max 10 MB)"})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Endpoints
#
# async def vs def - chosen on purpose:
#   - "/", "/health", "/model-info" only read values already in memory. They
#     never block, so `async def` runs them directly on the event loop with no
#     thread overhead.
#   - "/predict" is plain `def`. See the comment on that function.
# ---------------------------------------------------------------------------
@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        service="corn-leaf-disease-classifier",
        version=API_VERSION,
        description="POST a corn leaf image to /predict to get probabilities for all 3 classes.",
        endpoints={
            "GET /health": "liveness + readiness",
            "GET /model-info": "model version, classes and test metrics",
            "POST /predict": "multipart upload, field name 'file'",
            "GET /docs": "interactive API docs",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
)
async def health(request: Request):
    """
    200 when the model is loaded and ready, 503 when it isn't.

    WHY the status CODE matters, not just the JSON body: Docker's HEALTHCHECK,
    Kubernetes probes and load balancers only look at the code. Returning 200
    with {"model_loaded": false} would tell them "all good" while every
    prediction fails.
    """
    clf = get_classifier(request)
    if clf.is_loaded:
        return HealthResponse(status="healthy", model_loaded=True)
    body = HealthResponse(status="unhealthy", model_loaded=False, error=clf.error)
    return JSONResponse(status_code=503, content=body.model_dump())


@app.get("/model-info", response_model=ModelInfoResponse, responses={503: {"model": ErrorResponse}})
async def model_info(request: Request):
    clf = get_classifier(request)
    if not clf.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    m = clf.metrics
    training_date = m.get("training_date")
    return ModelInfoResponse(
        model_file=MODEL_FILE,
        model_version=f"{(training_date or 'unknown')[:10]}-{clf.sha256[:12]}",
        sha256=clf.sha256,
        classes=clf.class_names,
        input_size=[IMG_SIZE, IMG_SIZE, 3],
        training_date=training_date,
        test_accuracy=m.get("test_accuracy"),
        per_class=m.get("per_class"),
        n_test=m.get("n_test"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Not a valid image"},
        413: {"model": ErrorResponse, "description": "File over 10 MB"},
        503: {"model": ErrorResponse, "description": "Model not loaded yet"},
    },
)
def predict(request: Request, file: Annotated[UploadFile, File(description="A corn leaf photo (JPEG/PNG)")]):
    """
    WHY plain `def` and NOT `async def` (the opposite of most tutorials):

    Running the model is CPU-heavy, BLOCKING work - roughly 30-100 ms of pure
    computation with no `await` anywhere inside it. FastAPI runs plain `def`
    endpoints in a thread pool, so while one thread is crunching numbers, the
    event loop stays free to answer /health and other requests.

    If this were `async def`, the inference would run ON the event loop itself.
    Nothing else could run until it finished: health checks would time out and
    every other user would queue behind this one prediction. `async def` only
    helps when the function spends its time `await`-ing I/O (a database, an
    HTTP call). It does nothing for CPU work, and it actually hurts here.
    """
    clf = get_classifier(request)

    # 503, not 500: the client's request was fine, the server just isn't ready.
    # Load balancers treat 503 as "retry later or try another instance".
    if not clf.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded, try again shortly")

    # Read at most 1 byte more than the limit. If we got more than the limit,
    # the file is too big - and we never held more than 10 MB in memory.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # WHY decode with Pillow instead of trusting the Content-Type header or the
    # file extension: both are set by the client and can be anything. Actually
    # decoding the bytes is the only reliable "is this an image?" test.
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force a full decode now, so a truncated file fails HERE as a 400
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        logger.info("Rejected upload %r: not a decodable image", file.filename)
        # `from None`: this is an expected, handled case - don't chain Pillow's
        # internal error onto it.
        raise HTTPException(status_code=400, detail="File is not a valid image (JPEG or PNG expected)") from None

    start = time.perf_counter()
    probs = clf.predict(image)
    inference_ms = (time.perf_counter() - start) * 1000

    # Sort highest first so clients can display the list as-is.
    probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
    top_class, top_prob = next(iter(probs.items()))
    logger.info("Predicted %s (%.3f) for %r in %.0f ms", top_class, top_prob, file.filename, inference_ms)

    return PredictionResponse(
        predicted_class=top_class,
        confidence=top_prob,
        probabilities=probs,
        inference_ms=round(inference_ms, 1),
    )
