FROM python:3.13.15-slim AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git bubblewrap \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt requirements-dev.txt ./
COPY third_party/nexau/ third_party/nexau/
RUN python -m pip install --no-cache-dir -r requirements-dev.txt
COPY scripts/test_assets.py /opt/prepare_test_assets.py
ENV TIKTOKEN_CACHE_DIR=/opt/tokenizer_cache PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
WORKDIR /repo

FROM dependencies AS checks
# Explicit network preparation of public tokenizer data, not model inference.
RUN python /opt/prepare_test_assets.py --cache-dir /opt/tokenizer_cache --download
