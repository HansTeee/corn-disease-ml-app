# Single-container image for Hugging Face Spaces (Docker SDK).
# Runs BOTH processes: FastAPI on 127.0.0.1:8000 (internal) and Streamlit on
# 0.0.0.0:7860 (the only port Spaces routes to).
#
#   docker build -t ml-app .
#   docker run -p 7860:7860 ml-app        -> http://localhost:7860

# WHY python:3.13-slim:
# - slim (Debian, glibc), NOT alpine: Alpine uses musl libc, so TensorFlow and
#   numpy have no prebuilt wheels and must compile from source - very slow
#   builds and often a BIGGER image.
# - 3.13: the exact Python version the model was trained and tested with, and
#   numpy 2.5.4 requires Python >= 3.12 (3.11 can't install it).

# ============================================================================
# Stage 1: builder - install dependencies into a virtual environment
# ============================================================================
FROM python:3.13-slim AS builder

# Compilers, in case any package lacks a prebuilt wheel. They live only in this
# stage: the final image never contains them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# WHY a venv: it puts every installed package in one folder (/opt/venv) that we
# can copy into the final stage in a single step.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# WHY copy requirements.txt BEFORE any code: Docker caches each step and reuses
# it until something it depends on changes. Dependencies change rarely; code
# changes constantly. In this order, editing api/main.py reuses the cached
# (slow, ~300 MB) pip install. In the opposite order, every code edit would
# reinstall TensorFlow.
COPY requirements.txt .
RUN pip install -r requirements.txt

# ============================================================================
# Stage 2: runtime - a clean image with only the venv and the app
# ============================================================================
FROM python:3.13-slim

# PYTHONUNBUFFERED=1: print() and log lines go out immediately, so they appear
#   in `docker logs` in real time instead of sitting in a buffer (or being lost
#   if the container crashes).
# PYTHONDONTWRITEBYTECODE=1: don't litter the image with .pyc files.
# TF_CPP_MIN_LOG_LEVEL=2: hide TensorFlow's noisy C++ info messages.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user

# HF Spaces runs containers as a non-root user with UID 1000. We create that
# user here and make it own the app files. Otherwise everything "works on my
# machine" (as root) and then fails with permission errors after deploy, e.g.
# when Streamlit tries to write its config folder.
RUN useradd --create-home --uid 1000 user

COPY --from=builder /opt/venv /opt/venv

WORKDIR /home/user/app
# Least-often-changed first: the model (10 MB) rarely changes, code often does.
COPY --chown=user:user models/ ./models/
COPY --chown=user:user api/ ./api/
COPY --chown=user:user ui/ ./ui/
COPY --chown=user:user start.sh ./

# WHY the sed: if this repo is checked out on Windows, start.sh can get CRLF
# line endings, and Linux then fails with the baffling "/bin/sh^M: not found".
# Stripping \r makes the build immune to that.
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

USER user

# Documentation only (Spaces reads the port from README metadata), but it tells
# humans and tools which port matters. 8000 is NOT exposed: only Streamlit, in
# the same container, talks to the API, over localhost.
EXPOSE 7860

# WHY python instead of curl: python is already in the image; installing curl
# would add a package just for this one line.
# start-period: TensorFlow + model load takes several seconds; failures during
# this window don't count against the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["./start.sh"]
