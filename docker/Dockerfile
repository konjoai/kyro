# ── builder ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only in this stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── runtime ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user
RUN groupadd -r ragos && useradd -r -g ragos ragos

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY --chown=ragos:ragos ragos/ ./ragos/
COPY --chown=ragos:ragos pyproject.toml .

# Install the package itself (editable in dev; proper install here)
RUN pip install --no-cache-dir --no-deps .

USER ragos

EXPOSE 8000

# Health check — relies on the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["ragos", "serve", "--host", "0.0.0.0", "--port", "8000"]
