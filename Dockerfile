# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools needed by some packages (e.g., numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="Clovertex Data Engineering Pipeline"
LABEL description="Clinical and genomics data pipeline for Clovertex DE internship assignment"

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy source code and data
COPY pipeline/ ./pipeline/
COPY data/ ./data/

# Create datalake mount point; outputs are written here
# docker-compose mounts the host ./datalake directory to /app/datalake
# so outputs persist after the container exits
RUN mkdir -p datalake/raw datalake/refined datalake/consumption/plots

# Non-root user for security best-practice
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /tmp/matplotlib \
    && chown appuser:appuser /tmp/matplotlib
USER appuser

ENV MPLCONFIGDIR=/tmp/matplotlib

# The pipeline exits with code 0 on success, non-zero on failure
# docker-compose will report the exit code so CI can catch errors
CMD ["python", "-m", "pipeline.main"]