# interpretable-ml

A small toolkit and web service for model interpretation and local explanation methods. This repository contains implementations and orchestration for several interpretability techniques plus a lightweight web UI and a service entrypoint.

## Contents

Top-level files and folders you’ll commonly use:

- `service.py` — main HTTP service that exposes the explanation features (run this to start the app).
- `client.py` — a simple client example for interacting with the running service.
- `requirements.txt` — Python dependencies for development and running locally.
- `docker-compose.yml`, `Dockerfile`, `service.env` — Docker-based setup for containerized deployment.
- `input.json` — example input used by the project (keeps example requests or configuration).
- `tests.ipynb` — an exploratory notebook for manual tests and demonstrations.

Packages and purpose:

- `common/` — shared utilities, configuration and helpers used across modules.
- `model/` — model loading and model-related helpers.
- `lime/` — LIME implementation and structs.
- `counterfactual_explanations/` — counterfactual explanation code and helpers.
- `global_surrogate/` — global surrogate explanations.
- `permutation_feature_importance/` — permutation feature importance.
- `ppi/` — per-point importance implementation.
- `instances/` — utilities to format and validate input instances.
- `static/`, `templates/` — front-end assets and HTML templates for the web UI.

## Quick overview

This project provides programmatic components to compute model explanations and a small web UI to run methods interactively. You can run the service locally (with a virtual environment) or with Docker.

## Requirements

- Python 3.8+ (the project uses standard libraries and packages listed in `requirements.txt`).
- Docker & docker-compose (optional, required for containerized runs).

## Local development (recommended)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the service locally:

```bash
python service.py
```

4. Use the provided `client.py` to make example requests to the service or open the web UI served by `service.py` (it uses templates in `templates/` and assets in `static/`).

Tip: configuration values and secrets are stored in `service.env` — copy or edit it for local overrides if needed.

## Docker (quick start)

1. Build and start with docker-compose (from repo root):

```bash
docker-compose up --build
```

2. The service will be available at the address printed by the compose logs (commonly `http://localhost:5000` or the port configured in the environment file). Use `client.py` or the web UI templates to interact.

## Typical usage

- Prepare an instance using the helpers in `instances/` or by formatting JSON compatible with the model inputs (see `input.json` for an example).
- Use one of the explanation modules to compute an explanation for the instance, or call the HTTP API exposed by `service.py`.

The repository is structured so methods are modular — you can import and call them directly from Python scripts, or use the included orchestration layer in `orchestrator/` to run workflows that combine methods.

## Project structure (short)

- `service.py` — start server and route requests to orchestrator.
- `client.py` — example client demonstrating how to call the service programmatically.
- `orchestrator/` — pieces that coordinate method execution and return combined outputs.
- `model/` — loads `model.hdf5` (or other model artifacts) and exposes a model interface.

## Examples

- Run a local explanation: start `service.py`, then run `python client.py` (adjust the sample payload in `client.py` or `input.json`).
- Open `tests.ipynb` for interactive examples and demonstrations.

## Notes & troubleshooting

- If the service fails to start, check the Python dependencies with `pip list` and compare to `requirements.txt`.
- When using Docker, ensure your user has permission to run Docker commands or prefix them with `sudo`.
- Environment-specific settings are controlled via `service.env`; check that the environment variables referenced there are set or provided to the container.

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Create an issue describing the change or feature.
2. Make a branch off `main` for the work.
3. Add tests or a short notebook example (`tests.ipynb`) that demonstrates the change, when applicable.
4. Open a pull request describing the change.
