# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# System dependencies commonly needed by scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . /app

# Reasonable defaults; can be overridden in compose
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    MODEL_PATH=/workspace/material/model.hdf5 \
    PLOT_PATH=/workspace/outputs

EXPOSE 5000

# Default to gunicorn in the image; compose will override with dev server for reload
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "service:create_app()"]
