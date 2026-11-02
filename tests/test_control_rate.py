"""验证控制频率是精确的 30 Hz，且以物理步计的相位时长落在 tick 边界上。

第一版数据集要求采集帧率与真机推理循环一致（30 fps），所以 `real_safe` 的控制周期
必须精确等于 1/30 s（`sampling_hz == 30.0`）。这些时长还必须是 `physics_per_tick`
的整数倍，否则 `pick_cube.step()` 的断言会在采集时才失败，而不是在测试里。
"""

from __future__ import annotations

import pytest

from controllers.motion_profiles import MOTION_PROFILES, get_motion_profile
from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

PHASE_STEP_FIELDS = (
    "initial_physics_steps",
    "lift_hold_physics_steps",
    "release_settle_physics_steps",
)
# 相位时长改成 30 Hz 之前的值，按 0.002 s/步 定义；换算后的秒数要对得上它们。
LEGACY_PHYSICS_STEPS = {
    "simulation_fast": {
        "initial_physics_steps": 120,
        "lift_hold_physics_steps": 256,
        "release_settle_physics_steps": 200,
    },
    "real_safe": {
        "initial_physics_steps": 160,
        "lift_hold_physics_steps": 320,
        "release_settle_physics_steps": 240,
    },
}
LEGACY_TIMESTEP = 0.002


def test_scene_timestep_gives_exactly_30_hz_for_real_safe():
    model, _ = load_model(FIXED_SCENE_PATH)
    timestep = float(model.opt.timestep)
    assert timestep == 1.0 / 480.0, "场景物理步长应为 1/480 s"

    profile = get_motion_profile("real_safe")
    control_period = profile.physics_per_tick * timestep
    # 严格相等：timestep 的十进制写法就是为了让这里不出现 29.999… 这类浮点残差。
    assert control_period == 1.0 / 30.0
    assert 1.0 / control_period == 30.0


@pytest.mark.parametrize("name", sorted(MOTION_PROFILES))
def test_phase_physics_steps_are_whole_ticks(name):
    profile = get_motion_profile(name)
    for field in PHASE_STEP_FIELDS:
        value = getattr(profile, field)
        assert value % profile.physics_per_tick == 0, f"{name}.{field}={value} 不是整 tick"


@pytest.mark.parametrize("name", sorted(MOTION_PROFILES))
def test_phase_durations_keep_their_original_seconds(name):
    """时长按原秒数换算，误差不超过半个 tick —— 四舍五入到整 tick 的必然结果。"""
    model, _ = load_model(FIXED_SCENE_PATH)
    timestep = float(model.opt.timestep)
    profile = get_motion_profile(name)
    half_tick = profile.physics_per_tick * timestep / 2.0
    for field, legacy_steps in LEGACY_PHYSICS_STEPS[name].items():
        legacy_seconds = legacy_steps * LEGACY_TIMESTEP
        current_seconds = getattr(profile, field) * timestep
        assert abs(current_seconds - legacy_seconds) <= half_tick + 1e-12, (
            f"{name}.{field} 换算后 {current_seconds:.5f}s 偏离原时长 {legacy_seconds:.5f}s"
        )
