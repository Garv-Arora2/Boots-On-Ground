"""Project root and data directory paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = DATA / "assets"
SYNTHETIC = DATA / "synthetic"
SATELLITE = DATA / "satellite"
