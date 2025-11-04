# Permafrost Digital Twin (permafrost_DT)

This repository implements a digital twin system for permafrost thermal modeling and monitoring. The system combines classical finite difference methods (FDM) with physics-informed neural networks (PINNs) to simulate, infer, and visualize subsurface temperature fields.

# Contents
- [About this Project](#about-this-project)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running the Digital Twin](#running-the-digital-twin)
  - [Running Tests](#running-tests)
- [Services Overview](#services-overview)
- [Infrastructure Setup](#infrastructure-setup)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Contributing](#contributing)

# About this Project

**Goal:** The goal of this digital twin is to provide accurate thermal modeling and monitoring of permafrost, enabling better understanding and prediction of subsurface temperature dynamics.

**Key Features:**
- Real-time temperature field simulation using FDM
- Physics-informed neural networks for forward modeling and parameter inversion
- Boundary condition estimation from sensor data
- Visualization of temperature fields and model outputs
- Time-series data storage and analysis
- Message-based service architecture

The system uses InfluxDB for time-series storage and RabbitMQ for inter-service messaging.

# System Architecture

The digital twin follows a microservices architecture pattern, with distinct services communicating via RabbitMQ messaging:

## Core Services

1. **Observation Ingestion Server** (`digital_twin/monitoring/observation_ingestion`)
   - Ingests temperature sensor data
   - Writes observations to InfluxDB
   - Publishes data for other services

2. **Boundary Forcing Server** (`digital_twin/monitoring/boundary_forcing`)
   - Processes sensor data to derive boundary conditions
   - Supports synthetic boundary generation
   - Provides inputs for simulation models

3. **FDM Simulator** (`digital_twin/simulator/fdm`)
   - Classical finite difference method solver
   - Simulates subsurface temperature evolution
   - Writes results to InfluxDB

4. **PINN Forward Model** (`digital_twin/simulator/pinn_forward`)
   - Physics-informed neural network implementation
   - Predicts temperature fields from boundary conditions
   - Optional training mode for model refinement

5. **PINN Inversion Model** (`digital_twin/simulator/pinn_inversion`)
   - Infers soil parameters from temperature data
   - Uses FDM results for parameter estimation
   - Optional training mode for optimization

6. **Visualization Gateway** (`digital_twin/visualization/viz_gateway`)
   - Aggregates results from all models
   - Generates visualization-ready messages
   - Supports real-time data streaming

7. **Visualization Dashboard** (`digital_twin/visualization/viz_gateway/streamlit_viz_app.py`)
   - Streamlit-based web UI consuming the same InfluxDB streams
   - Provides contour plots, comparisons, and diagnostics
   - Offers auto-refresh and manual reload controls

## Infrastructure Components

- **Communication Layer** (`digital_twin/communication/`)
   - RabbitMQ messaging utilities
   - Message schemas and validation
   - Logging configuration

- **Data Access Layer** (`digital_twin/data_access/`)
   - InfluxDB connection helpers
   - Time-series data management
   - Docker deployment assets

## Repository Structure

```
permafrost_DT/
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── startup.conf           # Service configuration
├── pytest.ini            # Test configuration
├── data/                 # Data storage
├── docs/                 # Architecture diagrams
├── integration_tests/    # System-level tests
├── logs/                 # Service logs
├── resources/            # Additional resources
└── software/            # Main codebase
    └── digital_twin/    # Core services
        ├── communication/  # Messaging
        ├── data_access/   # Database
        ├── monitoring/    # Data ingestion
        ├── simulator/     # Models
        └── visualization/ # Viz gateway
```

# Prerequisites

Required software:
- Python 3.10+
- pip
- Docker & Docker Compose
- Git

Optional but recommended:
- CUDA-capable GPU for PINN acceleration
- Python virtual environment (venv)

**Note on PyTorch:** The installation method depends on your OS and whether you want CUDA support. Visit https://pytorch.org/get-started/locally/ to select the appropriate installation command. The `requirements.txt` includes a CPU-only `torch` entry by default.

# Getting Started

## Installation

1. Clone the repository:
```bash
git clone https://github.com/simplyadil/permafrost_DT.git
cd permafrost_DT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install GPU-enabled PyTorch:
```bash
# Visit pytorch.org and follow instructions for your system
```

## Configuration

### Environment Variables

The services use these environment variables (with defaults):

```bash
# InfluxDB Configuration
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=<your-token>
INFLUX_ORG=permafrost
INFLUX_BUCKET=permafrost

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_USER=permafrost
RABBITMQ_PASSWORD=permafrost
```

Set these in your shell or create a `.env` file.

### Infrastructure Setup

1. Start InfluxDB:
```bash
cd software/digital_twin/data_access/influxdbserver
docker-compose up -d
```

2. Start RabbitMQ:
```bash
cd software/digital_twin/communication/installation/rabbitmq
docker-compose up -d
```

Helper scripts are available in `software/startup/docker_services/`:
```bash
python software/startup/docker_services/start_influxdb.py
python software/startup/docker_services/start_rabbitmq.py
```

## Running individual services

Many services are written as simple Python modules you can run directly. Example: run the viz gateway (ensure env vars are set):

```bash
python software/digital_twin/visualization/viz_gateway/viz_gateway_server.py
```

Launch the interactive dashboard (optional, non-blocking) to visualise aggregated outputs:

```bash
streamlit run software/digital_twin/visualization/viz_gateway/streamlit_viz_app.py
```

The command honours the `viz_dashboard` block in `startup.conf` (host, port, refresh, history depth). You can still run `python -m software.startup.start_viz_dashboard` if you prefer the managed launcher—it simply spawns the Streamlit app with the configured settings. The UI auto-refreshes every 60 seconds, offers a **Reload Data** button, and swaps in friendly placeholders until upstream datasets or diagnostics are available.

If you run scripts directly from the `software` tree, the tests and some scripts include a small `sys.path` adjustment to allow running modules directly for development.

### Synthetic observation stream

When you do not have physical sensors connected, the startup scripts can recycle the FDM simulator output as a sensor feed. By default `synthetic_observations.enabled` is `true` in `startup.conf`, which makes the FDM server publish depth samples (0–5 m) to the `permafrost.record.sensors.state` queue so the observation ingestion service keeps progressing. Adjust `synthetic_observations.depths_m` (integers 0–5) to change which depths are emitted, or disable the block entirely once real hardware is available.

### Synthetic boundary forcing

The boundary forcing server now falls back to a sinusoidal surface temperature when InfluxDB has no recent observations. Control this via the `synthetic_boundary` section in `startup.conf` (enabled by default). Tweak `start_day` and `step_days` to match the cadence you need, or set `enabled` to `false` once real surface measurements are being ingested.

### PINN forward model

The PINN forward service can operate purely in inference mode. Place your trained checkpoint at `software/models/pinn_forward/freezing_soil_pinn.pt` (or update `pinn_forward.model_path` in `startup.conf`) and leave `pinn_forward.enable_training` set to `false` to skip the costly retraining loop. Flip the flag back to `true` only when you intentionally want the service to optimise the network at runtime.

### PINN inversion model

Similarly, the inversion service can emit precomputed soil parameters without running the heavy optimisation loop. Drop your trained weights at `software/models/pinn_inversion/inversion_pinn.pt` (or point `pinn_inversion.model_path` elsewhere) and keep `pinn_inversion.enable_training` set to `false`. Re-enable it only when you plan to re-fit the inversion PINN from incoming data. The inversion step now consumes the finite-difference simulation results directly, so you can run it alongside—or instead of—the forward PINN without chaining the two together.

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

# Troubleshooting

## Common Issues

### Connection Problems
- **RabbitMQ Connection Failed**
  - Verify `RABBITMQ_HOST` is correct
  - Check container status: `docker ps | grep rabbitmq`
  - Check management UI: http://localhost:15672 
  - Default credentials: `permafrost`/`permafrost`

- **InfluxDB Not Reachable**
  - Verify `INFLUX_URL` is correct
  - Check health endpoint: http://localhost:8086/health
  - Verify container is running: `docker ps | grep influxdb`
  - Check token validity in management UI

### Service Issues
- **PINN Model Loading Errors**
  - Verify model files exist in correct locations
  - Check PyTorch CUDA compatibility
  - Confirm Python environment has correct dependencies

- **Import Errors**
  - Run scripts from repository root
  - Use module notation: `python -m software.digital_twin.{service}`
  - Verify virtual environment is activated
