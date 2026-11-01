"""A bounded proportional visual-servoing baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection


@dataclass(frozen=True)
class ControllerConfig:
    image_width: int = 640
    desired_area: float = 1_800.0
    max_linear_speed: float = 0.30
    max_angular_speed: float = 1.20
    horizontal_gain: float = 1.5
    distance_gain: float = 0.00035


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_command(
    detection: Detection | None,
    config: ControllerConfig | None = None,
) -> tuple[float, float]:
    """Return ``(linear_x, angular_z)`` for a detected target.

    The command is deliberately simple: horizontal image error controls yaw;
    target area controls forward speed. If the target is missing, the robot
    stops instead of pretending it has a valid measurement.
    """

    config = config or ControllerConfig()
    if detection is None:
        return 0.0, 0.0

    image_center_x = config.image_width / 2.0
    normalized_error = (detection.center_x - image_center_x) / image_center_x
    angular_z = clamp(
        -config.horizontal_gain * normalized_error,
        -config.max_angular_speed,
        config.max_angular_speed,
    )

    area_error = config.desired_area - detection.area
    linear_x = clamp(
        config.distance_gain * area_error,
        0.0,
        config.max_linear_speed,
    )
    return linear_x, angular_z

