"""Start the visualization dashboard (Streamlit)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from software.startup.utils.config import load_startup_config


def main() -> None:
    config = load_startup_config()
    viz_cfg = config.get("viz_dashboard", {})
    host = str(viz_cfg.get("host", "0.0.0.0"))
    port = int(viz_cfg.get("port", 8501))

    app_path = (
        Path(__file__).resolve().parents[1]
        / "digital_twin"
        / "visualization"
        / "viz_dashboard"
        / "viz_dashboard_app.py"
    )
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
