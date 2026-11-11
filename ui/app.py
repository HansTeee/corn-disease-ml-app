"""
Streamlit frontend. It talks to the FastAPI backend over HTTP and nothing else.

THE KEY RULE OF THIS FILE: no TensorFlow, no model, no inference here.
The UI is just another client of the API, like curl or a mobile app would be.
That separation means:
  - the model can be retrained, replaced or moved to a bigger machine without
    touching this file;
  - the API can serve other clients too;
  - this process stays small (no TensorFlow in memory) and starts instantly.

Run locally (API must already be running on port 8000):
    streamlit run ui/app.py
"""
import os

import requests
import streamlit as st

# WHY an environment variable: one setting covers all three deployments.
#   - local dev:          default, API on this machine
#   - single container:   default too, both processes share localhost
#   - split deployment:   set API_URL=http://api:8000 (docker-compose) or a real URL
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

PREDICT_TIMEOUT_S = 30  # long enough for a slow CPU, short enough that users aren't left hanging
INFO_TIMEOUT_S = 5      # health/model-info are trivial; if they're slow, the API is in trouble
LOW_CONFIDENCE = 0.6

# The API returns the dataset's folder names. These are for humans.
DISPLAY_NAMES = {
    "Corn_(maize)___Common_rust_": "Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}


def pretty(class_name: str) -> str:
    # .get with a fallback: if the model ever gains a class, the UI still works.
    return DISPLAY_NAMES.get(class_name, class_name)


st.set_page_config(page_title="Corn Leaf Disease Classifier", page_icon="🌽", layout="centered")


# ---------------------------------------------------------------------------
# API helpers. Each returns (data, error_message) so the UI code below never
# has to deal with exceptions or raw tracebacks.
# ---------------------------------------------------------------------------
UNREACHABLE = f"Can't reach the prediction API at {API_URL}. Is the backend running?"


@st.cache_data(ttl=10, show_spinner=False)
def get_json(path: str):
    """
    GET a small JSON endpoint.

    WHY cache for 10 s: Streamlit reruns the whole script on every click. Without
    a cache, every button press would also re-fetch /health and /model-info.
    """
    try:
        r = requests.get(f"{API_URL}{path}", timeout=INFO_TIMEOUT_S)
    except requests.ConnectionError:
        return None, UNREACHABLE
    except requests.Timeout:
        return None, f"The API took longer than {INFO_TIMEOUT_S}s to answer {path}."
    if r.status_code == 503:
        # /health sends a JSON body with the reason even when unhealthy.
        try:
            return r.json(), "Model is still loading or failed to load."
        except ValueError:
            return None, "Model is still loading or failed to load."
    if not r.ok:
        return None, f"API returned HTTP {r.status_code} for {path}."
    return r.json(), None


def call_predict(filename: str, data: bytes, mime: str):
    """POST the image as multipart form data, exactly like `curl -F file=@leaf.jpg`."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, data, mime or "application/octet-stream")},
            timeout=PREDICT_TIMEOUT_S,
        )
    except requests.ConnectionError:
        return None, UNREACHABLE
    except requests.Timeout:
        return None, (
            f"The prediction took longer than {PREDICT_TIMEOUT_S}s and was cancelled. "
            "The server may be overloaded - try again."
        )

    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 503:
        return None, "The model is still loading. Try again in a few seconds."
    if r.status_code in (400, 413, 422):
        # These are the client's fault, and the API's message says what to fix.
        try:
            return None, r.json().get("detail", "The API rejected this file.")
        except ValueError:
            return None, "The API rejected this file."
    return None, f"Something went wrong on the server (HTTP {r.status_code}). Please try again."


# ---------------------------------------------------------------------------
# Sidebar: makes the frontend/backend split visible
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Backend status")
    st.caption(f"API: `{API_URL}`")

    health, health_err = get_json("/health")
    if health and health.get("model_loaded"):
        st.success("API healthy · model loaded")
    elif health:
        st.error("API is up but the model is NOT loaded")
        if health.get("error"):
            st.caption(health["error"])
    else:
        st.error(health_err)

    if st.button("Refresh status"):
        get_json.clear()  # drop the cache so we really re-check
        st.rerun()

    info, _ = get_json("/model-info") if health and health.get("model_loaded") else (None, None)
    if info:
        st.header("Model")
        st.markdown(
            f"**Version:** `{info['model_version']}`  \n"
            f"**Test accuracy:** {info['test_accuracy']:.1%} on {info['n_test']} held-out images  \n"
            f"**Trained:** {(info.get('training_date') or '?')[:10]}"
        )
        st.markdown("**Classes:** " + ", ".join(pretty(c) for c in info["classes"]))
        if info.get("per_class"):
            with st.expander("Per-class metrics"):
                st.table(
                    {
                        name: {k: round(v, 3) for k, v in m.items() if k != "support"}
                        for name, m in info["per_class"].items()
                    }
                )
        st.caption(f"[API docs]({API_URL}/docs)")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🌽 Corn Leaf Disease Classifier")
st.write(
    "Upload a photo of a single corn (maize) leaf. The model returns how likely it is "
    "to be **Common rust**, **Northern Leaf Blight** or **Healthy**."
)
st.caption(
    "This page doesn't run the model itself. It sends your image to a separate "
    "FastAPI service and displays what comes back."
)

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# Forget the previous result when a different file is uploaded, so we never
# show image B next to image A's prediction.
file_key = (uploaded.name, uploaded.size) if uploaded else None
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.result = None
    st.session_state.error = None

if uploaded:
    st.image(uploaded, caption=uploaded.name, width=320)

    if st.button("Predict", type="primary"):
        with st.spinner("Asking the API..."):
            result, error = call_predict(uploaded.name, uploaded.getvalue(), uploaded.type)
        st.session_state.result, st.session_state.error = result, error

    if st.session_state.error:
        st.error(st.session_state.error)
        # The retry button just repeats the same request with the same file.
        if st.button("Retry"):
            get_json.clear()
            with st.spinner("Retrying..."):
                result, error = call_predict(uploaded.name, uploaded.getvalue(), uploaded.type)
            st.session_state.result, st.session_state.error = result, error
            st.rerun()

    result = st.session_state.result
    if result:
        top, conf = result["predicted_class"], result["confidence"]
        st.subheader(f"Prediction: {pretty(top)}")
        st.metric("Confidence", f"{conf:.1%}")

        # The model can't say "I don't know" - it always splits 100% across
        # three classes, even for a photo of a cat. So the UI must say it.
        if conf < LOW_CONFIDENCE:
            st.warning(
                f"⚠️ Low confidence ({conf:.0%}). The model isn't sure. This may not be a "
                "corn leaf, or the photo may be blurry, dark or show several leaves."
            )

        # ALL three classes, not just the winner: 51/49 and 99/1 are very
        # different answers that would look identical as a single label.
        st.write("**All class probabilities**")
        for name, p in result["probabilities"].items():
            st.progress(p, text=f"{pretty(name)}: {p:.1%}")

        st.caption(
            f"Inference took {result['inference_ms']:.0f} ms on the server. "
            "High confidence does not guarantee the image is a corn leaf: the model only "
            "knows these three classes."
        )
else:
    st.info("Upload an image to get started.")
