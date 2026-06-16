"""CLI: generate before/after demo assets in data/assets/."""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boots_on_ground.demo_assets import create_after, create_before  # noqa: E402


def main():
    print("before:", create_before())
    print("after:", create_after())


if __name__ == "__main__":
    main()
