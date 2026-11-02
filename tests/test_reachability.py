"""验证执行前的可达性预检。

预检的作用是在真正执行之前发现注定失败的样本：探测中 (0.20, ±0.12) 可以完成，
(0.25, ±0.12) 会在抬升阶段 IK 发散。
"""

from __future__ import annotations

import numpy as np
import pytest
from module_loader import load_script

PICK_CUBE_SCRIPT = "scripts/pick_cube.py"


@pytest.fixture(scope="module")
def module():
    return load_script(PICK_CUBE_SCRIPT, "pick_cube_reachability")


def make_task(module, xy, yaw_deg=37.0):
    return module.TaskSample(
        cube_xy=xy,
        cube_yaw=float(np.deg2rad(yaw_deg)),
        box_xy=module.SCENE_BOX_XY,
        seed=0,
    )


@pytest.mark.parametrize(
    "xy",
    [
        (0.1732, -0.1000),  # 极坐标 r=0.20, θ=-30°
        (0.1732, 0.1000),  # r=0.20, θ=+30°
        (0.1905, 0.1100),  # r=0.22, θ=+30°
        (0.2078, 0.1200),  # r=0.24, θ=+30°
        (0.2078, -0.1200),  # r=0.24, θ=-30°
        (0.2200, 0.0),  # r=0.22, θ=0
        (0.2400, 0.0),  # r=0.24, θ=0
    ],
)
def test_known_good_points_are_reachable(module, xy):
    report = module.check_task_reachable(make_task(module, xy))
    assert report.reachable, f"{xy} 应判定可达，实际 {report.max_error_m * 1000:.2f} mm"
    assert report.failing_phase is None


@pytest.mark.parametrize("xy", [(0.25, 0.12), (0.25, -0.12)])
def test_known_bad_points_are_rejected(module, xy):
    report = module.check_task_reachable(make_task(module, xy))
    assert not report.reachable, f"{xy} 应判定不可达"
    assert report.failing_phase == "抬升"
    assert report.max_error_m > module.REACHABILITY_TOLERANCE_M["抬升"]


@pytest.mark.parametrize("xy", [(0.16, 0.0), (0.20, 0.0), (0.19, 0.02)])
def test_points_inside_the_folded_arm_are_rejected(module, xy):
    """folded_ready 的夹爪停在桌面 x≈0.13–0.19、y≈0，方块采样到这里会被弹开。"""
    report = module.check_task_reachable(make_task(module, xy))
    assert not report.reachable, f"{xy} 与折叠臂穿模，应判定不可达"
    assert report.initial_collision
    assert report.failing_phase == "初始状态"


def test_grasp_critical_segments_keep_strict_tolerance(module):
    """接近/下降/盒内必须用 5 mm 硬标准，否则预检形同虚设。"""
    limits = module.REACHABILITY_TOLERANCE_M
    assert limits["接近"] == 0.005
    assert limits["下降"] == 0.005
    assert limits["下降到盒内"] == 0.005
    # 中间路点放宽到 25 mm，但仍低于 move_to 的 30 mm 失败判定。
    assert limits["抬升"] < 0.030
    assert limits["移动到盒子上方"] < 0.030


def test_custom_tolerance_changes_verdict(module):
    task = make_task(module, (0.22, 0.0))
    strict = dict.fromkeys(module.REACHABILITY_TOLERANCE_M, 1e-4)
    assert not module.check_task_reachable(task, tolerances_m=strict).reachable
    loose = dict.fromkeys(module.REACHABILITY_TOLERANCE_M, 1.0)
    assert module.check_task_reachable(task, tolerances_m=loose).reachable


def test_initial_collision_is_not_overridden_by_loose_tolerance(module):
    """穿模是几何问题，放宽 IK 误差上限也不应该放行。"""
    task = make_task(module, (0.20, 0.0))
    loose = dict.fromkeys(module.REACHABILITY_TOLERANCE_M, 1.0)
    report = module.check_task_reachable(task, tolerances_m=loose)
    assert not report.reachable
    assert report.initial_collision


def test_reachability_check_does_not_require_execution(module):
    """预检只解 IK，不推进物理时间。"""
    task = make_task(module, (0.20, 0.0))
    model, data = module.load_model(module.FIXED_SCENE_PATH)
    before = data.time
    module.check_task_reachable(task)
    assert data.time == before
    del model
