## Permafrost Digital Twin (permafrost_DT)

This repository contains a digital twin pipeline for permafrost thermal modeling. The project includes services that simulate, infer, and visualize subsurface temperature fields using a mix of classical (FDM) and physics-informed neural networks (PINNs). It uses InfluxDB for time-series storage and RabbitMQ for messaging between services.

In this README you'll find the repository layout, dependencies, environment variables, how to run services, and how to run tests.

---

## Quick project overview

- Purpose: run a pipeline that generates synthetic/observed temperature data, simulates via an FDM solver, runs PINN forward/inversion, stores results in InfluxDB, and publishes visualization-ready messages.
- Main components:
  - digital_twin/monitoring/observation_ingestion — observation ingestion and writing to InfluxDB
  - digital_twin/monitoring/boundary_forcing — derives boundary conditions for the simulator
  - digital_twin/simulator/fdm — FDM-based simulator that writes results to InfluxDB
  - digital_twin/simulator/pinn_forward — PINN forward model (PyTorch)
  - digital_twin/simulator/pinn_inversion — PINN inversion model (PyTorch)
  - digital_twin/visualization/viz_gateway — aggregates results and publishes visualization messages
  - digital_twin/communication — shared messaging utilities and schemas
  - digital_twin/data_access — InfluxDB helpers and Docker assets

## Repo layout (important files)

```
requirements.txt
README.md
data/
docs/                 # diagrams and architecture drawings (.drawio)
software/
  digital_twin/
    communication/     # messaging, logger, schemas, rabbitmq assets
    data_access/       # influx helper + docker assets
    monitoring/        # ingestion + boundary forcing servers
    simulator/         # FDM + PINN servers
    visualization/     # viz gateway server
  startup/             # orchestration scripts
integration_tests/     # lightweight behavioural checks
logs/
```

## Prerequisites

- Python 3.10+ (venv recommended)
- pip
- Docker & docker-compose (optional, for local InfluxDB and RabbitMQ)

Notes on PyTorch: PyTorch installation often depends on your OS, Python version and whether you want CUDA support. Install PyTorch following the selector on https://pytorch.org/get-started/locally/ (the `requirements.txt` contains a `torch` entry but not a pinned wheel).

## Steps for Setup

1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

2. Install the Python dependencies

```bash
pip install -r requirements.txt
```

If you need GPU-enabled PyTorch, install `torch` separately using the official instructions.

## Environment variables

The services read configuration from environment variables and/or code defaults. Key variables:

- INFLUX_URL — e.g. `http://localhost:8086`
- INFLUX_TOKEN — InfluxDB API token
- INFLUX_ORG — InfluxDB org name
- INFLUX_BUCKET — bucket used by the services
- RABBITMQ_HOST — hostname for RabbitMQ (default: `localhost`)
- RABBITMQ_USER — (default: `permafrost`)
- RABBITMQ_PASSWORD — (default: `permafrost`)

Set these in your shell or an env file before running services that connect to InfluxDB/RabbitMQ.

## Starting local infrastructure (Docker)

Docker Compose manifests are available under the digital twin package:

- InfluxDB: `software/digital_twin/data_access/influxdbserver/docker-compose.yml`
- RabbitMQ: `software/digital_twin/communication/installation/rabbitmq/docker-compose.yml`

To start InfluxDB:

```bash
cd software/digital_twin/data_access/influxdbserver
docker-compose up -d
# Wait for health endpoint (script in software/startup/docker_services can help)
```

To start RabbitMQ:

```bash
cd software/digital_twin/communication/installation/rabbitmq
docker-compose up -d
```

There are helper scripts in `software/startup/docker_services/` such as `start_influxdb.py` and `start_rabbitmq.py` which perform a compose up and poll the health endpoints. You can run these from the repo root (with a Python venv active).

## Running individual services

Many services are written as simple Python modules you can run directly. Example: run the viz gateway (ensure env vars are set):

```bash
python software/digital_twin/visualization/viz_gateway/viz_gateway_server.py
```

If you run scripts directly from the `software` tree, the tests and some scripts include a small `sys.path` adjustment to allow running modules directly for development.

## Tests

Integration-style smoke tests live in the top-level `integration_tests/` directory.

Run tests (simple invocation):

```bash
pytest -q
```

Or you can run individual test scripts directly:

```bash
python integration_tests/test_viz_gateway_server.py
python integration_tests/test_observation_ingestion_server.py
```

## JSON schemas and messaging

- Message schemas live in `software/digital_twin/communication/schemas/`.
- Messaging uses RabbitMQ via `pika`. `software/digital_twin/communication/messaging.py` exposes `RabbitMQClient` with convenience helpers (publish, consume). Messages are validated against JSON Schema before publishing.

## Development notes and tips

- Pin PyTorch carefully for reproducible installs; use the PyTorch website to pick the right wheel for CUDA or CPU.
- The codebase uses pandas and numpy extensively — ensure `pandas` and `numpy` are installed in the venv.
- If you hit import errors like `ModuleNotFoundError: No module named 'digital_twin'`, make sure you run scripts from the repository root or use `python -m software.digital_twin.visualization.viz_gateway.viz_gateway_server` style imports.

## Troubleshooting

- RabbitMQ connection issues: check `RABBITMQ_HOST` and that the container/service is running (`docker ps`). The management UI (if enabled) is usually on port 15672.
- InfluxDB not reachable: check `INFLUX_URL` and the health endpoint at `/health` (default port 8086).
- PyTorch import errors: verify your installed `torch` wheel is compatible with your Python version and OS.

## Contributing

- Please open issues/PRs on the repository for fixes or feature additions.
- Keep dependency updates minimal and run tests after upgrades.

## License & attribution

This repo does not currently include a license file; please add a LICENSE file if you plan to open-source or publish.

---

If you'd like, I can:
- pin exact versions from your active venv (`pip freeze` -> populate `requirements.txt`) for reproducibility,
- add a small `dev` Makefile or `scripts/` to automate starting Docker infra and tests,
- or add a short `CONTRIBUTING.md` to capture common development steps.
