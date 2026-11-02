"""验证官方处理器原样使用及 MuJoCo 单位适配，不初始化设备或窗口。"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

from so_ws.teleoperators.official_gamepad_mujoco import JOINTS, OfficialGamepadMujoco


@pytest.fixture(autouse=True)
def no_devices(monkeypatch):
    import hid
    import mujoco.viewer
    import pygame
    import serial

    def forbidden(*args, **kwargs):
        raise AssertionError("官方链路验证不允许连接设备或 GUI")

    for obj, name in [
        (hid, "device"),
        (serial, "Serial"),
        (pygame, "init"),
        (pygame.joystick, "init"),
        (mujoco.viewer, "launch_passive"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)


def packet(*, enabled=True, xyz=(0.0, 0.0, 0.0), gripper=1):
    return {
        "enabled": enabled,
        "action": {
            "delta_x": xyz[0],
            "delta_y": xyz[1],
            "delta_z": xyz[2],
            "gripper": gripper,
        },
    }


def test_official_pipeline_classes_and_defaults():
    sim = OfficialGamepadMujoco()
    assert [type(step).__name__ for step in sim.pipeline.steps] == [
        "MapDeltaActionToRobotActionStep",
        "EEReferenceAndDelta",
        "EEBoundsAndSafety",
        "GripperVelocityToJoint",
        "InverseKinematicsRLStep",
    ]
    assert all(type(step).__module__.startswith("lerobot.") for step in sim.pipeline.steps)
    mapper, reference, bounds, gripper, inverse = sim.pipeline.steps
    assert mapper.position_scale == 1.0 and mapper.noise_threshold == 1e-3
    assert reference.end_effector_step_sizes == {"x": 0.02, "y": 0.02, "z": 0.02}
    assert not reference.use_latched_reference and reference.use_ik_solution
    assert bounds.end_effector_bounds == {"min": [-1.0] * 3, "max": [1.0] * 3}
    assert bounds.max_ee_step_m == EEBoundsAndSafety.__dataclass_fields__["max_ee_step_m"].default
    assert bounds.raise_on_jump == EEBoundsAndSafety.__dataclass_fields__["raise_on_jump"].default
    assert gripper.speed_factor == 1.0 and gripper.discrete_gripper and gripper.clip_max == 100
    assert not inverse.initial_guess_current_joints
    signature = inspect.signature(sim.kinematics.inverse_kinematics)
    assert signature.parameters["position_weight"].default == 1.0
    assert signature.parameters["orientation_weight"].default == 0.01


def test_task_parameters_match_official_examples():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/gamepad_official.json").read_text())
    vendor = root / "third_party/lerobot_v061"
    document = (vendor / "docs/source/hilserl.mdx").read_text()
    examples = []
    for block in re.findall(r"```json\s*\n(.*?)```", document, re.DOTALL):
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        ik = value.get("env", {}).get("processor", {}).get("inverse_kinematics", {})
        if "end_effector_step_sizes" in ik:
            examples.append(ik["end_effector_step_sizes"])
    assert config["inverse_kinematics"]["end_effector_step_sizes"] in examples
    tree = ast.parse((vendor / "examples/phone_to_so100/teleoperate.py").read_text())
    bounds = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "EEBoundsAndSafety"
        ):
            bounds.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == "end_effector_bounds")
    assert config["inverse_kinematics"]["end_effector_bounds"] in bounds


def test_original_urdf_tcp_and_native_scene_initial_state():
    sim = OfficialGamepadMujoco()
    assert sim.kinematics.target_frame_name == "gripper_frame_link"
    assert sim.kinematics.joint_names == list(JOINTS)
    expected = sim.model.key_qpos[sim.model.key("folded_ready").id]
    np.testing.assert_array_equal(sim.data.qpos, expected)
    assert sim.data.time == 0
    root = Path(__file__).resolve().parents[1]
    original = root / "models/so101/so101_new_calib.urdf"
    assert "gamepad_tcp" not in original.read_text()


def test_one_original_ik_call_per_input_without_overrides(monkeypatch):
    sim = OfficialGamepadMujoco()
    calls = []
    original = sim.kinematics.inverse_kinematics

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim.kinematics, "inverse_kinematics", tracked)
    sim.receive(packet(xyz=(0.03, 0.0, 0.0)), 0)
    sim.advance(0)
    assert calls == [{}]
    target = sim.target.copy()
    sim.advance(10)
    assert calls == [{}]
    np.testing.assert_array_equal(sim.target, target)


def test_no_extra_2mm_rejection_or_rearm_gate(monkeypatch):
    sim = OfficialGamepadMujoco()
    monkeypatch.setattr(sim.kinematics, "inverse_kinematics", lambda seed, desired: seed.copy())
    sim.receive(packet(xyz=(1, 0, 0)), 0)
    sim.advance(0)
    assert sim.enabled
    assert sim.reason == "官方处理链控制中"
    assert not hasattr(sim, "can_enable")
    assert np.isfinite(sim.target).all()


def test_official_workspace_clamps_and_accepts_inward_motion():
    sim = OfficialGamepadMujoco()
    bounds = sim.pipeline.steps[2]
    action = {"ee.x": 1.2, "ee.y": 0, "ee.z": 0, "ee.wx": 0, "ee.wy": 0, "ee.wz": 0, "ee.gripper_vel": 1}
    assert bounds.action(action.copy())["ee.x"] == 1.0
    action["ee.x"] = 0.98
    assert bounds.action(action.copy())["ee.x"] == 0.98


def test_official_jump_handling_is_not_overridden():
    sim = OfficialGamepadMujoco()
    bounds = sim.pipeline.steps[2]
    action = {"ee.x": 0, "ee.y": 0, "ee.z": 0, "ee.wx": 0, "ee.wy": 0, "ee.wz": 0, "ee.gripper_vel": 1}
    bounds.action(action.copy())
    action["ee.x"] = 0.2
    with pytest.raises(ValueError, match="EE jump"):
        bounds.action(action.copy())


@pytest.mark.parametrize("command,expected_end", [(0, "low"), (2, "high")])
def test_official_gripper_semantics_map_to_mujoco(command, expected_end):
    sim = OfficialGamepadMujoco()
    sim.receive(packet(gripper=command), 0)
    sim.advance(0)
    expected = sim.gripper_range[0 if expected_end == "low" else 1]
    assert sim.target[5] == pytest.approx(expected)


def test_original_model_files_not_written():
    root = Path(__file__).resolve().parents[1] / "models/so101"
    paths = list(root.glob("*.urdf")) + list(root.glob("*.xml"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    sim = OfficialGamepadMujoco()
    sim.receive(packet(xyz=(0.02, 0.0, 0.0)), 0)
    sim.advance(0)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


def test_official_chain_runs_physics_without_legacy_task_box():
    sim = OfficialGamepadMujoco()
    for index in range(15):
        sim.receive(packet(xyz=(0.02, 0.0, 0.03)), index * sim.dt)
        sim.advance(index * sim.dt)
    assert sim.data.time == pytest.approx(15 / 30)
    assert sim.enabled
    assert np.isfinite(sim.data.qpos).all()
