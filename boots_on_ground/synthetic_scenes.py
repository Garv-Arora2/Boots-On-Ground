"""Procedural 512×512 demo scenes for offline use."""

import os

import cv2
import numpy as np
from PIL import Image

from boots_on_ground.paths import SYNTHETIC

SIZE = 512

BARE = (210, 180, 140)
VEG = (34, 139, 34)
WATER = (30, 144, 255)
BUILDING = (120, 70, 40)
ROAD = (128, 128, 128)


def _canvas():
    img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = BARE
    return img


def _noise(img, std=6):
    n = np.random.default_rng(0).normal(0, std, img.shape)
    return np.clip(img.astype("float32") + n, 0, 255).astype(np.uint8)


def _buildings(img, rects):
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), BUILDING, -1)


def urban_grid():
    img = _canvas()
    for p in (128, 256, 384):
        cv2.line(img, (p, 0), (p, SIZE - 1), ROAD, 7)
        cv2.line(img, (0, p), (SIZE - 1, p), ROAD, 7)
    rects = []
    for cx in (64, 192, 320, 448):
        for cy in (64, 192, 320, 448):
            rects += [(cx - 30, cy - 22, 28, 20), (cx + 2, cy - 22, 28, 20),
                      (cx - 30, cy + 4, 28, 20), (cx + 2, cy + 4, 28, 20)]
    _buildings(img, rects)
    cv2.rectangle(img, (170, 170), (242, 242), VEG, -1)
    cv2.circle(img, (320, 70), 16, WATER, -1)
    return img


def river_crossing():
    img = _canvas()
    river = np.array([[180, 0], [240, 120], [200, 250], [260, 380], [220, 512],
                      [330, 512], [300, 380], [350, 250], [310, 120], [300, 0]], np.int32)
    cv2.fillPoly(img, [river], WATER)
    cv2.line(img, (0, 256), (511, 256), ROAD, 9)
    cv2.line(img, (256, 256), (256, 60), ROAD, 7)
    _buildings(img, [(60, 200, 34, 26), (100, 210, 30, 26), (60, 250, 36, 24)])
    cv2.circle(img, (420, 380), 70, VEG, -1)
    return img


def forest_clearing():
    img = _canvas()
    for (cx, cy, r) in [(120, 120, 110), (380, 150, 120), (160, 400, 130), (400, 410, 110)]:
        cv2.circle(img, (cx, cy), r, VEG, -1)
    pts = np.array([[20, 480], [120, 360], [240, 380], [320, 240], [460, 180]], np.int32)
    cv2.polylines(img, [pts], False, ROAD, 8)
    lake = np.array([[300, 300], [380, 290], [410, 350], [350, 400], [295, 360]], np.int32)
    cv2.fillPoly(img, [lake], WATER)
    return img


def coastal_town():
    img = _canvas()
    sea = np.array([[0, 0], [170, 0], [150, 130], [200, 260], [150, 380], [190, 512],
                    [0, 512]], np.int32)
    cv2.fillPoly(img, [sea], WATER)
    cv2.line(img, (210, 0), (250, 512), ROAD, 8)
    cv2.line(img, (250, 256), (511, 256), ROAD, 7)
    _buildings(img, [(300, 120, 34, 26), (350, 130, 30, 24), (300, 300, 32, 26),
                     (370, 300, 30, 28), (420, 150, 34, 24), (430, 320, 28, 26)])
    cv2.circle(img, (430, 430), 60, VEG, -1)
    return img


def desert_outpost():
    img = _canvas()
    cv2.line(img, (0, 380), (511, 300), ROAD, 7)
    cv2.line(img, (256, 512), (300, 120), ROAD, 6)
    _buildings(img, [(280, 110, 34, 28), (320, 120, 30, 26), (285, 150, 36, 24),
                     (330, 160, 28, 26)])
    cv2.circle(img, (130, 160), 46, VEG, -1)
    cv2.circle(img, (130, 160), 20, WATER, -1)
    return img


def mountain_valley():
    img = _canvas()
    cv2.fillPoly(img, [np.array([[0, 0], [512, 0], [512, 120], [0, 220]], np.int32)], VEG)
    cv2.fillPoly(img, [np.array([[0, 512], [512, 512], [512, 360], [0, 430]], np.int32)], VEG)
    river = np.array([[0, 250], [150, 270], [300, 250], [512, 280],
                      [512, 320], [300, 300], [150, 320], [0, 300]], np.int32)
    cv2.fillPoly(img, [river], WATER)
    cv2.line(img, (40, 200), (470, 360), ROAD, 7)
    cv2.line(img, (256, 0), (256, 511), ROAD, 7)
    _buildings(img, [(80, 180, 30, 24), (430, 360, 30, 26)])
    return img


def mixed_terrain():
    img = _canvas()
    cv2.circle(img, (95, 95), 70, VEG, -1)
    cv2.circle(img, (420, 430), 85, VEG, -1)
    cv2.fillPoly(img, [np.array([[10, 380], [120, 360], [175, 440], [110, 505], [10, 505]],
                                np.int32)], WATER)
    _buildings(img, [(330, 40, 42, 30), (400, 60, 36, 36), (350, 115, 52, 30), (445, 125, 30, 42)])
    cv2.line(img, (0, 256), (511, 256), ROAD, 8)
    cv2.line(img, (256, 0), (256, 511), ROAD, 8)
    cv2.line(img, (0, 0), (511, 511), ROAD, 7)
    return img


SCENES = {
    "Urban grid": ("urban_grid.png", urban_grid),
    "River crossing": ("river_crossing.png", river_crossing),
    "Forest & lake": ("forest_clearing.png", forest_clearing),
    "Coastal town": ("coastal_town.png", coastal_town),
    "Desert outpost": ("desert_outpost.png", desert_outpost),
    "Mountain valley": ("mountain_valley.png", mountain_valley),
    "Mixed terrain": ("mixed_terrain.png", mixed_terrain),
}


def scene_path(label: str) -> str:
    return str(SYNTHETIC / SCENES[label][0])


def ensure_scene(label: str) -> str:
    """Generate one scene if missing; return its file path."""
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    path = scene_path(label)
    if not os.path.exists(path):
        img = _noise(SCENES[label][1]())
        Image.fromarray(img).save(path)
    return path


def ensure_all() -> list:
    return [ensure_scene(label) for label in SCENES]
