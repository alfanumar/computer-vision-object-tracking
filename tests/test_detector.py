import cv2
import numpy as np

from vision_demo.detector import detect_orange_target
from vision_demo.synthetic_scene import SceneConfig, render_frame, target_position


def test_synthetic_target_is_detected_near_expected_center():
    config = SceneConfig()
    frame_index = 10
    detection, mask = detect_orange_target(render_frame(frame_index, config))
    expected_x, expected_y = target_position(frame_index, config)

    assert detection is not None
    assert mask.shape == (config.height, config.width)
    assert abs(detection.center_x - expected_x) <= 3
    assert abs(detection.center_y - expected_y) <= 3


def test_detector_returns_none_when_target_is_absent():
    frame = np.full((360, 640, 3), 32, dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (180, 55, 55), -1)

    detection, _ = detect_orange_target(frame)

    assert detection is None

