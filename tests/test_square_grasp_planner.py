import numpy as np

from controllers.square_grasp_planner import (
    minimum_jerk,
    select_square_grasp_yaw,
    square_grasp_yaw_candidates,
)


def test_square_yaw_candidates_and_selection():
    candidates = square_grasp_yaw_candidates(0.0)
    assert len(candidates) == 4
    assert np.allclose(sorted(abs(yaw) for yaw in candidates), [0.0, np.pi / 2, np.pi / 2, np.pi])
    assert select_square_grasp_yaw(0.0, 0.1) == candidates[0]


def test_minimum_jerk_endpoints():
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0


def test_selection_compares_against_current_world_yaw():
    """第二个参数是夹爪闭合方向的世界 yaw，已经包含 shoulder_pan 的贡献。"""
    # 世界 yaw 接近 0 时选 0° 候选，接近 90° 时改选 90° 候选。
    assert np.isclose(select_square_grasp_yaw(0.0, 0.0), 0.0)
    assert np.isclose(select_square_grasp_yaw(0.0, 1.4), np.pi / 2)
    assert np.isclose(select_square_grasp_yaw(0.0, -1.4), -np.pi / 2)
