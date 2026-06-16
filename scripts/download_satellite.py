"""CLI: download real satellite imagery into data/satellite/."""

import argparse
import os
import sys

# Allow running as: python scripts/download_satellite.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boots_on_ground.imagery_download import download_all, download_scene  # noqa: E402
from config.satellite_scenes import REAL_DIR, REAL_SCENES, scene_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Download satellite scenes for BOG")
    parser.add_argument("--list", action="store_true", help="List catalog and download status")
    parser.add_argument("--only", action="append", help="Download only this scene (repeatable)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    if args.list:
        for label, meta in REAL_SCENES.items():
            status = "downloaded" if os.path.isfile(scene_path(label)) else "missing"
            print(f"  [{status}] {label} — {meta['note']}")
        return

    os.makedirs(REAL_DIR, exist_ok=True)
    if args.only:
        for label in args.only:
            download_scene(label, force=args.force)
    else:
        download_all(force=args.force)


if __name__ == "__main__":
    main()
