"""SO-101 正方体抓取的轻量规划工具。"""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def square_grasp_yaw_candidates(base_yaw: float) -> tuple[float, ...]:
    """生成正方体绕 Z 轴的四个等效抓取方向。"""
    return tuple(wrap_to_pi(base_yaw + index * np.pi / 2.0) for index in range(4))


def select_square_grasp_yaw(base_yaw: float, current_wrist_yaw: float) -> float:
    """选取与当前腕部旋转距离最小的等效方向。"""
    candidates = square_grasp_yaw_candidates(base_yaw)
    return min(candidates, key=lambda yaw: abs(wrap_to_pi(yaw - current_wrist_yaw)))


def minimum_jerk(progress: float) -> float:
    """返回 0 到 1 之间的五次 minimum-jerk 进度。"""
    t = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
