"""Classical HSV segmentation and contour-based target detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Detected target geometry in pixel coordinates."""

    center_x: int
    center_y: int
    width: int
    height: int
    area: float


def detect_orange_target(frame_bgr: np.ndarray) -> tuple[Detection | None, np.ndarray]:
    """Detect the orange target and return both detection and binary mask.

    This is intentionally a transparent, classical CV baseline rather than a
    trained object-detection model. The HSV thresholds are suitable for the
    synthetic scene and should be calibrated for a real camera.
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 80, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50.0:
        return None, mask

    x, y, width, height = cv2.boundingRect(contour)
    return Detection(
        center_x=x + width // 2,
        center_y=y + height // 2,
        width=width,
        height=height,
        area=float(area),
    ), mask


def annotate_frame(
    frame_bgr: np.ndarray,
    detection: Detection | None,
    command: tuple[float, float],
    target_center_x: int,
) -> np.ndarray:
    """Draw detection and controller output without changing the input."""

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    cv2.line(annotated, (target_center_x, 0), (target_center_x, height), (255, 255, 255), 1)

    if detection is not None:
        left = detection.center_x - detection.width // 2
        top = detection.center_y - detection.height // 2
        cv2.rectangle(
            annotated,
            (left, top),
            (left + detection.width, top + detection.height),
            (0, 255, 0),
            2,
        )
        cv2.circle(annotated, (detection.center_x, detection.center_y), 4, (0, 255, 0), -1)
        status = f"target=({detection.center_x},{detection.center_y})"
    else:
        status = "target=not detected"

    linear_x, angular_z = command
    cv2.putText(annotated, status, (16, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(
        annotated,
        f"cmd: linear.x={linear_x:.2f}, angular.z={angular_z:.2f}",
        (16, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )
    return annotated

