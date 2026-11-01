"""Synthetic camera scene used when physical camera data is unavailable."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    """Image and target parameters for the reproducible scene."""

    width: int = 640
    height: int = 360
    target_radius: int = 24
    target_bgr: tuple[int, int, int] = (0, 140, 255)


def target_position(frame_index: int, config: SceneConfig) -> tuple[int, int]:
    """Return a deterministic target path that crosses the image center."""

    x = int(config.width * (0.18 + 0.64 * (0.5 + 0.5 * np.sin(frame_index * 0.07))))
    y = int(config.height * (0.50 + 0.20 * np.sin(frame_index * 0.11 + 0.8)))
    return x, y


def render_frame(frame_index: int, config: SceneConfig | None = None) -> np.ndarray:
    """Render one synthetic BGR camera frame with a target and distractors."""

    config = config or SceneConfig()
    frame = np.full((config.height, config.width, 3), 32, dtype=np.uint8)

    # A simple floor/grid gives the result a camera-like scene without hiding
    # that the target is generated data.
    for y in range(40, config.height, 40):
        cv2.line(frame, (0, y), (config.width, y), (48, 48, 48), 1)
    for x in range(40, config.width, 40):
        cv2.line(frame, (x, 0), (x, config.height), (48, 48, 48), 1)

    # Distractors use colors outside the target HSV range.
    cv2.rectangle(frame, (55, 70), (135, 130), (180, 55, 55), -1)
    cv2.rectangle(frame, (500, 210), (590, 315), (55, 180, 55), -1)

    center = target_position(frame_index, config)
    cv2.circle(frame, center, config.target_radius, config.target_bgr, -1)
    cv2.circle(frame, center, config.target_radius + 2, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "synthetic camera scene",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame

