"""CLI: generate all synthetic demo scenes into data/synthetic/."""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boots_on_ground.synthetic_scenes import ensure_all  # noqa: E402


def main():
    for path in ensure_all():
        print("wrote", path)


if __name__ == "__main__":
    main()
