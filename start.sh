#!/bin/sh
# Starts both processes inside the single container.
#   1. FastAPI (uvicorn) in the BACKGROUND on 127.0.0.1:8000 - internal only
#   2. wait until the API actually answers (no blind `sleep 10`)
#   3. Streamlit in the FOREGROUND on 0.0.0.0:7860 - the port HF Spaces routes to
set -e

echo "[start.sh] starting FastAPI on 127.0.0.1:8000"
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# WHY poll instead of sleep: model load time varies with the machine (a few
# seconds locally, longer on a busy free-tier host). A fixed sleep is either
# too short (UI starts, first requests fail) or wastes time. Polling waits
# exactly as long as needed.
#
# uvicorn only starts accepting connections AFTER the lifespan (model load)
# finishes, so ANY HTTP answer means startup is done:
#   exit 0 -> 200, model loaded
#   exit 2 -> 503, the API is up but the model failed to load
#   exit 1 -> no answer yet, keep waiting
check_api() {
    python - <<'EOF'
import sys, urllib.request, urllib.error
try:
    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
    sys.exit(0)
except urllib.error.HTTPError:
    sys.exit(2)
except Exception:
    sys.exit(1)
EOF
}

waited=0
while true; do
    status=0
    check_api || status=$?
    if [ "$status" -eq 0 ]; then
        echo "[start.sh] API healthy after ${waited}s"
        break
    elif [ "$status" -eq 2 ]; then
        # Start the UI anyway: its sidebar shows the problem, and /health keeps
        # reporting 503 for the orchestrator. Better than a silent restart loop.
        echo "[start.sh] WARNING: API is up but the model failed to load - see the log above"
        break
    fi
    # If uvicorn itself crashed (e.g. import error), waiting longer is pointless.
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[start.sh] ERROR: FastAPI process exited during startup"
        exit 1
    fi
    if [ "$waited" -ge 180 ]; then
        echo "[start.sh] ERROR: API did not answer within 180s"
        exit 1
    fi
    sleep 1
    waited=$((waited + 1))
done

echo "[start.sh] starting Streamlit on 0.0.0.0:7860"
# exec: Streamlit replaces this shell, so it receives Docker's stop signal
#   directly, and when Streamlit exits the container exits (and gets restarted)
#   instead of lingering half-alive.
# --server.address=0.0.0.0: Streamlit's default 127.0.0.1 only accepts
#   connections from INSIDE the container, so `-p 7860:7860` would seem to do nothing.
exec streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false
