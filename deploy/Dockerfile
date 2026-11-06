# Steelbound — container image.
# Local-first AI core; the HTTP API binds loopback only by design
# (see src/steelbound/server.py). This image keeps that binding inside
# the container — the Compose file publishes the port to the host's
# 127.0.0.1 only, never to a public interface.
# Build from the repository root:  docker build -f deploy/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package exactly as a pip user would (pyproject.toml /
# setuptools backend, packages under src/, console.html and model_data
# checkpoints carried as package data).
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir . \
    && rm -rf /app/src /app/pyproject.toml

# Non-root runtime user and the persistent data directory.
# The data dir holds the SQLite ledger database and workspace files;
# keep it on a volume (see docker-compose.yml) or all memory and
# ledger history is lost with the container.
RUN groupadd --system steelbound \
    && useradd --system --gid steelbound --no-create-home steelbound \
    && mkdir -p /data \
    && chown steelbound:steelbound /data

USER steelbound
VOLUME ["/data"]
EXPOSE 8765

# Real CLI (src/steelbound/cli.py): global --data-dir comes before the
# subcommand; `serve` accepts --host (default 127.0.0.1, loopback only —
# server.py refuses any non-loopback host) and --port (default 8765).
CMD ["steelbound", "--data-dir", "/data", "serve", "--host", "127.0.0.1", "--port", "8765"]
