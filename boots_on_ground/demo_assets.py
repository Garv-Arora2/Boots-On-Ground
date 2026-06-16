"""Bundled before/after images for the change-detection demo."""

import os

import cv2
import numpy as np
from PIL import Image

from boots_on_ground.paths import ASSETS

SIZE = 512
SAMPLE_BEFORE = ASSETS / "sample_before.png"
SAMPLE_AFTER = ASSETS / "sample_after.png"
SATELLITE_BEFORE = ASSETS / "satellite_before.png"
SATELLITE_AFTER = ASSETS / "satellite_after.png"

BARE = (210, 180, 140)
VEG = (34, 139, 34)
WATER = (30, 144, 255)
BUILDING = (120, 70, 40)
ROAD = (128, 128, 128)


def _base_scene(size: int = SIZE) -> np.ndarray:
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = BARE
    cv2.circle(img, (95, 95), 70, VEG, -1)
    cv2.circle(img, (420, 430), 85, VEG, -1)
    water_pts = np.array([[10, 380], [120, 360], [175, 440], [110, 505], [10, 505]], np.int32)
    cv2.fillPoly(img, [water_pts], WATER)
    for (x, y, w, h) in [(330, 40, 42, 30), (400, 60, 36, 36), (350, 115, 52, 30), (445, 125, 30, 42)]:
        cv2.rectangle(img, (x, y), (x + w, y + h), BUILDING, -1)
    cv2.line(img, (0, 256), (511, 256), ROAD, 8)
    cv2.line(img, (256, 0), (256, 511), ROAD, 8)
    cv2.line(img, (0, 0), (511, 511), ROAD, 7)
    return img


def create_before(path=None, size: int = SIZE) -> str:
    path = str(path or SAMPLE_BEFORE)
    ASSETS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_base_scene(size)).save(path)
    return path


def create_after(path=None, size: int = SIZE) -> str:
    path = str(path or SAMPLE_AFTER)
    img = _base_scene(size)
    for (x, y, w, h) in [(150, 300, 44, 34), (205, 305, 38, 38), (160, 360, 50, 32), (220, 360, 34, 44)]:
        cv2.rectangle(img, (x, y), (x + w, y + h), BUILDING, -1)
    cv2.circle(img, (420, 430), 85, BARE, -1)
    cv2.circle(img, (420, 430), 45, VEG, -1)
    cv2.line(img, (256, 330), (150, 330), ROAD, 7)
    cv2.fillPoly(img, [np.array([[10, 460], [110, 470], [110, 505], [10, 505]], np.int32)], BARE)
    ASSETS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def ensure_before(path=None) -> str:
    path = str(path or SAMPLE_BEFORE)
    if not os.path.exists(path):
        create_before(path)
    return path


def ensure_after(path=None) -> str:
    path = str(path or SAMPLE_AFTER)
    if not os.path.exists(path):
        create_after(path)
    return path


def ensure_satellite_pair() -> tuple[str, str]:
    """Bundled real satellite before/after pair (structure damage demo)."""
    before = str(SATELLITE_BEFORE)
    after = str(SATELLITE_AFTER)
    if not os.path.exists(before) or not os.path.exists(after):
        raise FileNotFoundError(
            "Satellite demo images missing from data/assets "
            "(satellite_before.png, satellite_after.png)."
        )
    return before, after
