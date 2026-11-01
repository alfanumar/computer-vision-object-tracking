from vision_demo.controller import ControllerConfig, compute_command
from vision_demo.detector import Detection


def test_missing_target_stops_robot():
    assert compute_command(None) == (0.0, 0.0)


def test_target_to_the_right_requests_left_turn():
    command = compute_command(Detection(560, 180, 48, 48, 1800.0))
    assert command[0] == 0.0
    assert command[1] < 0.0


def test_command_is_bounded():
    config = ControllerConfig()
    command = compute_command(Detection(640, 180, 10, 10, 1.0), config)
    assert 0.0 <= command[0] <= config.max_linear_speed
    assert -config.max_angular_speed <= command[1] <= config.max_angular_speed

