"""用开环回放证明 (obs, action) 严格同拍。

采集时 observer 在执行动作之前被调用，记录的是 (s_t, a_t)；a_t 会在第 t 个控制 tick
内线性插值执行。本测试把记录下来的 action 序列原样回放，要求每个 tick 开始时的
qpos 与记录的 qpos[t] 完全一致。只要动作错了一拍（例如记录成 a_{t-1}），回放轨迹
会立刻偏离，测试必然失败。
"""

from __future__ import annotations

import numpy as np
import pytest
from module_loader import load_script

from controllers.motion_profiles import get_motion_profile
from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

PICK_CUBE_SCRIPT = "scripts/pick_cube.py"
REPLAY_ATOL = 1e-9


def load_pick_cube(name: str):
    return load_script(PICK_CUBE_SCRIPT, name)


def record_episode(module, profile_name: str = "real_safe"):
    """运行一次专家演示，按 tick 记录 (qpos, action, phase)。"""
    trace = {"qpos": [], "qvel": [], "action": [], "phase": []}

    def observer(_model, data, phase, action):
        trace["qpos"].append(data.qpos[:6].copy())
        trace["qvel"].append(data.qvel[:6].copy())
        trace["action"].append(action[:6].copy())
        trace["phase"].append(phase)

    result = module.run(with_viewer=False, observer=observer, profile_name=profile_name)
    return result, trace


@pytest.mark.parametrize("profile_name", ["real_safe", "simulation_fast"])
def test_open_loop_replay_matches_recorded_states(profile_name):
    module = load_pick_cube(f"pick_cube_alignment_{profile_name}")
    result, trace = record_episode(module, profile_name)
    assert result.success, "专家演示本身必须成功，否则无法验证对齐"

    actions = np.asarray(trace["action"])
    states = np.asarray(trace["qpos"])
    # 初始状态那一帧必须被记录，且采集不止一帧。
    assert len(actions) == len(states) > 100

    model, data = load_model(FIXED_SCENE_PATH)
    module.configure_profile(get_motion_profile(profile_name), model.opt.timestep)
    module.reset_to_ready(model, data)

    assert np.allclose(data.qpos[:6], states[0], atol=REPLAY_ATOL, rtol=0.0), (
        "回放起点必须与采集第一帧完全一致"
    )

    for index, action in enumerate(actions):
        assert np.allclose(data.qpos[:6], states[index], atol=REPLAY_ATOL, rtol=0.0), (
            f"第 {index} 帧（phase={trace['phase'][index]}）对齐失败："
            f"回放 qpos 与记录 qpos 偏差 {np.abs(data.qpos[:6] - states[index]).max():.3e}。"
            "这通常说明观测和动作错了一拍。"
        )
        module.advance_to_ctrl(model, data, None, action, phase=trace["phase"][index])


def test_action_is_the_command_actually_applied():
    """记录的动作必须是限速后真正执行的 ctrl，而不是 IK 原始输出。"""
    module = load_pick_cube("pick_cube_alignment_semantics")
    _, trace = record_episode(module, "real_safe")
    actions = np.asarray(trace["action"])
    states = np.asarray(trace["qpos"])

    # 相邻 tick 的关节命令变化不能超过限速上限。
    profile = get_motion_profile("real_safe")
    assert module.ACTIVE_PROFILE is not None
    max_delta = np.concatenate(
        (
            np.full(5, profile.max_joint_velocity_rad_s * module.CONTROL_PERIOD),
            [profile.max_gripper_velocity_rad_s * module.CONTROL_PERIOD],
        )
    )
    # 第 0 帧的参考命令是 reset_to_ready 预置的 ctrl。
    model, data = load_model(FIXED_SCENE_PATH)
    module.configure_profile(profile, model.opt.timestep)
    module.reset_to_ready(model, data)
    previous = data.ctrl[:6].copy()
    for index, action in enumerate(actions):
        assert np.all(np.abs(action - previous) <= max_delta + 1e-12), (
            f"第 {index} 帧的动作超过单 tick 限速，说明记录的不是限速后的命令"
        )
        previous = action

    # 执行阶段确实在动：动作序列不是常量。
    assert np.abs(np.diff(actions, axis=0)).max() > 1e-3
    assert np.ptp(states[:, 0]) > 0.1
