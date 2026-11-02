"""验证俯视抓取的 yaw 对齐。

SO-101 手臂有 5 个自由度：3 个位置约束 + 1 个朝下约束之后，剩下的 wrist_roll 用来
控制夹爪闭合方向的 yaw。本测试在固定位置 (0.25, 0) 扫描方块 yaw，要求全部抓取成功；
同时验证不施加 yaw 约束时同一批任务无法对齐，说明这项约束不是可有可无的。
"""

from __future__ import annotations

import numpy as np
import pytest
from module_loader import load_script

from controllers.motion_profiles import get_motion_profile
from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

PICK_CUBE_SCRIPT = "scripts/pick_cube.py"
CUBE_YAW_GRID_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
# 固定位置：场景 keyframe 里方块中心就是 (0.25, 0, 0.015)。
FIXED_CUBE_XY = (0.25, 0.0)


def load_pick_cube(name: str):
    return load_script(PICK_CUBE_SCRIPT, name)


def fixed_yaw_task(module, yaw_deg: float):
    """固定位置 (0.25, 0)、指定方块 yaw 的任务。"""
    return module.TaskSample(
        cube_xy=FIXED_CUBE_XY,
        cube_yaw=float(np.deg2rad(yaw_deg)),
        box_xy=module.SCENE_BOX_XY,
        seed=0,
    )


@pytest.mark.parametrize("yaw_deg", CUBE_YAW_GRID_DEG)
def test_topdown_grasp_succeeds_for_yaw_grid(yaw_deg):
    module = load_pick_cube(f"pick_cube_yaw_{yaw_deg:.0f}")
    task = fixed_yaw_task(module, yaw_deg)
    assert module.run(with_viewer=False, profile_name="real_safe", task=task).success, (
        f"方块 yaw={yaw_deg}° 时完整演示失败"
    )


def _grasp_pose_yaw_error(module, target_yaw: float | None) -> tuple[float, float]:
    """在方块上方 12 mm 处求解 IK，返回 (yaw 误差, 位置误差)。"""
    model, data = load_model(FIXED_SCENE_PATH)
    module.configure_profile(get_motion_profile("real_safe"), model.opt.timestep)
    module.reset_to_ready(model, data)
    gripper_id = module.body_id(model, module.GRIPPER_BODY)
    cube_id = module.body_id(model, module.CUBE_BODY)
    assert np.allclose(data.xpos[cube_id][:2], FIXED_CUBE_XY, atol=1e-9)
    qpos_ids, dof_ids = module.arm_addresses(model)
    target = data.xpos[cube_id].copy() + np.array([0.0, 0.0, 0.012])
    q, position_error = module.solve_ik(
        model, data, target, qpos_ids, dof_ids, gripper_id, target_yaw=target_yaw
    )
    data.qpos[qpos_ids] = q
    module.mujoco.mj_forward(model, data)
    achieved = module.gripper_yaw(model, data, gripper_id)
    return abs(float(module.wrap_to_pi(target_yaw - achieved))), float(position_error)


@pytest.mark.parametrize("yaw_deg", CUBE_YAW_GRID_DEG)
def test_yaw_constraint_pins_gripper_orientation(yaw_deg):
    module = load_pick_cube(f"pick_cube_yaw_ik_{yaw_deg:.0f}")
    target_yaw = float(np.deg2rad(yaw_deg))
    yaw_error, position_error = _grasp_pose_yaw_error(module, target_yaw)
    assert yaw_error < 0.01, f"target_yaw={yaw_deg}° 时夹爪 yaw 误差 {np.rad2deg(yaw_error):.3f}°"
    assert position_error < 0.005


def test_without_yaw_constraint_orientation_is_uncontrolled():
    """没有 yaw 约束时 wrist_roll 由位置/朝下约束决定，无法跟随方块 yaw。"""
    module = load_pick_cube("pick_cube_yaw_ik_free")
    achieved = []
    for yaw_deg in CUBE_YAW_GRID_DEG:
        model, data = load_model(FIXED_SCENE_PATH)
        module.configure_profile(get_motion_profile("real_safe"), model.opt.timestep)
        module.reset_to_ready(model, data)
        gripper_id = module.body_id(model, module.GRIPPER_BODY)
        cube_id = module.body_id(model, module.CUBE_BODY)
        qpos_ids, dof_ids = module.arm_addresses(model)
        target = data.xpos[cube_id].copy() + np.array([0.0, 0.0, 0.012])
        q, _ = module.solve_ik(model, data, target, qpos_ids, dof_ids, gripper_id)
        data.qpos[qpos_ids] = q
        module.mujoco.mj_forward(model, data)
        achieved.append(module.gripper_yaw(model, data, gripper_id))
    assert np.ptp(achieved) < 1e-3, "无约束 IK 的夹爪 yaw 不应随方块 yaw 变化"
    largest_error = max(
        abs(float(module.wrap_to_pi(np.deg2rad(deg) - achieved_yaw)))
        for deg, achieved_yaw in zip(CUBE_YAW_GRID_DEG, achieved, strict=True)
    )
    assert largest_error > np.deg2rad(10.0), "无约束 IK 本来就应该对不齐，否则本测试失去意义"


# 采样范围的四个角 + yaw=67.5°→90° 这一段。标定网格的 yaw 只取到 67.5°，而采样范围是
# [0°, 90°)：正方体在 90° 处与 0° 几何等价，但 yaw 对齐误差随姿态变化，所以角落上接近
# 90° 的样本必须单独覆盖，不能靠 67.5° 和 90° 的等价性推断。
CORNER_RADII_M = (0.20, 0.24)
CORNER_AZIMUTHS_DEG = (-30.0, 30.0)
HIGH_YAW_GRID_DEG = (80.0, 89.0)


@pytest.mark.parametrize("radius_m", CORNER_RADII_M)
@pytest.mark.parametrize("azimuth_deg", CORNER_AZIMUTHS_DEG)
@pytest.mark.parametrize("yaw_deg", HIGH_YAW_GRID_DEG)
def test_sampling_range_corners_succeed_for_high_yaw(radius_m, azimuth_deg, yaw_deg):
    """采样范围 (0.20, 0.24) × ±30° 的四个角上，yaw 接近 90° 时完整演示必须成功。"""
    module = load_pick_cube(f"pick_cube_corner_{radius_m:.2f}_{azimuth_deg:+.0f}_{yaw_deg:.0f}")
    azimuth = float(np.deg2rad(azimuth_deg))
    task = module.TaskSample(
        cube_xy=(radius_m * float(np.cos(azimuth)), radius_m * float(np.sin(azimuth))),
        cube_yaw=float(np.deg2rad(yaw_deg)),
        box_xy=module.SCENE_BOX_XY,
        seed=0,
    )
    result = module.run(with_viewer=False, profile_name="real_safe", task=task)
    assert result.success, (
        f"r={radius_m} m θ={azimuth_deg:+.0f}° yaw={yaw_deg}° 时完整演示失败："
        f"{result.failure_phase} / {result.failure_reason}"
    )


@pytest.mark.parametrize("yaw_deg", HIGH_YAW_GRID_DEG)
def test_high_yaw_constraint_pins_gripper_orientation(yaw_deg):
    """yaw 接近 90° 时 IK 的 yaw 约束同样要能锁定夹爪闭合方向。"""
    module = load_pick_cube(f"pick_cube_high_yaw_ik_{yaw_deg:.0f}")
    target_yaw = float(np.deg2rad(yaw_deg))
    yaw_error, position_error = _grasp_pose_yaw_error(module, target_yaw)
    assert yaw_error < 0.01, f"target_yaw={yaw_deg}° 时夹爪 yaw 误差 {np.rad2deg(yaw_error):.3f}°"
    assert position_error < 0.005
