from __future__ import annotations

import argparse
from pathlib import Path

from itds.runner import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="itds", description="Insider Threat Detection System (student-scale)")
    parser.add_argument("--config", default="configs/itds.yml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_pipeline(config_path)
    return 0
