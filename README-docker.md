# Running interpretable-ml in Docker

This repo includes a simple Flask API in `service.py` that exposes the orchestrator.

## What this setup provides
- Reproducible environment via Dockerfile
- Live code reload in the container (compose runs `python service.py` with `FLASK_DEBUG=1`)
- Mounted model and outputs directories so no rebuild is needed when you change code or data

## Files
- `Dockerfile`: Builds a Python 3.11 image with all deps
- `docker-compose.yml`: Starts the API with live mounts
- `.dockerignore`: Keeps the image small
- `requirements.txt`: Python dependencies

## Start the API
```bash
# From inside the interpretable-ml folder
docker compose up --build
```

Then visit:
- Health check: http://localhost:5001/health

## Posting an explanation request
The API now accepts file paths only (no large arrays in the request):

```json
{
  "input_path": "/absolute/path/to/input.json",
  "instances_path": "/absolute/path/to/instances.hdf5",
  "ground_truth_path": "/absolute/path/to/labels.csv"  
}
```

- `input.json` must follow the structure in the repo root example (method_name, labels, instances_config, model_config, method_config).
- `instances.hdf5` should contain a dataset (preferably named `tracings`, `instances`, `data`, or `X`).
- `labels.csv` should have a single column of labels or a column named `label`, `labels`, `y`, or `target`.

## Volumes and paths
- Host `../material` is mounted to `/workspace/material` (read-only)
- Host `../outputs` is mounted to `/workspace/outputs`
- Environment variables `MODEL_PATH` and `PLOT_PATH` point to those

Adjust mounts in `docker-compose.yml` if your host paths differ.
