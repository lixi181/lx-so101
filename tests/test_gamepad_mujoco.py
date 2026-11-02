"""用官方接口的假手柄和无窗口 MuJoCo 验证适配，不初始化 SDL 或任何硬件。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig, gamepad_utils
from lerobot.teleoperators.utils import TeleopEvents
from module_loader import load_script

from so_ws.teleoperators.gamepad_mujoco import INPUT_TIMEOUT, JOINT_SPEED, GamepadMujoco


@pytest.fixture(autouse=True)
def forbid_devices(monkeypatch):
    import mujoco.viewer
    import serial

    def forbidden(*args, **kwargs):
        raise AssertionError("离线测试禁止启动 SDL、Viewer 或串口")

    monkeypatch.setattr(serial, "Serial", forbidden)
    monkeypatch.setattr(gamepad_utils.hid, "device", forbidden)
    monkeypatch.setattr(gamepad_utils.pygame, "init", forbidden)
    monkeypatch.setattr(gamepad_utils.pygame.joystick, "init", forbidden)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", forbidden)


@pytest.fixture
def official(tmp_path, monkeypatch):
    pad = gamepad_utils.GamepadController()
    axes = [0.0, 0.0, 0.0, 0.0]
    buttons = [False] * 8
    events = []
    pad.joystick = SimpleNamespace(
        get_axis=lambda index: axes[index], get_button=lambda index: buttons[index]
    )
    monkeypatch.setattr(
        gamepad_utils.pygame.event, "get", lambda: [events.pop(0) for _ in range(len(events))]
    )
    teleop = GamepadTeleop(GamepadTeleopConfig(id="offline", calibration_dir=tmp_path))
    teleop.gamepad = pad
    yield SimpleNamespace(teleop=teleop, axes=axes, buttons=buttons, events=events)
    # 测试从未初始化 SDL，不调用官方 stop() 的真实清理路径。
    teleop.gamepad = None


def packet(seq, now, *, enabled=True, xyz=(0.0, 0.0, 0.0), gripper=1):
    return {
        "seq": seq,
        "stamp": now,
        "enabled": enabled,
        "action": dict(zip(("delta_x", "delta_y", "delta_z", "gripper"), (*xyz, gripper), strict=True)),
    }


def arm(sim):
    sim.receive(packet(0, 0.0, enabled=False), 0.0)
    sim.receive(packet(1, 0.0), 0.0)
    assert sim.enabled


def test_dependency_uses_moved_source():
    import lerobot

    root = Path(__file__).resolve().parents[1]
    assert Path(lerobot.__file__).resolve().is_relative_to(root / "third_party/lerobot_v061")


def test_official_axes_deadzone_gripper_and_intervention(official):
    official.axes[:] = [0.4, -0.5, 0.0, -0.3]
    official.buttons[5] = True
    official.events.append(SimpleNamespace(type=gamepad_utils.pygame.JOYBUTTONDOWN, button=6))
    action = official.teleop.get_action()
    np.testing.assert_allclose([action[f"delta_{axis}"] for axis in "xyz"], [0.5, -0.4, 0.3])
    assert action["gripper"] == 0
    assert official.teleop.get_teleop_events()[TeleopEvents.IS_INTERVENTION]
    official.events.append(SimpleNamespace(type=gamepad_utils.pygame.JOYBUTTONUP, button=6))
    official.axes[:] = [0.05, -0.03, 0.0, 0.09]
    action = official.teleop.get_action()
    assert [action[f"delta_{axis}"] for axis in "xyz"] == [0, 0, 0]
    assert action["gripper"] == 1


def test_initial_pose_and_paused_hold():
    sim = GamepadMujoco()
    np.testing.assert_allclose(sim.goal_xyz, [0.22, 0, 0.08], atol=0.002)
    target = sim.target.copy()
    for _ in range(60):
        sim.advance(0.0)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    assert sim.data.time == pytest.approx(2.0)


def test_rb_requires_neutral_release_before_first_motion():
    sim = GamepadMujoco()
    sim.receive(packet(0, 0.0, xyz=(1, 0, 0)), 0.0)
    assert not sim.enabled
    sim.receive(packet(1, 0.0, enabled=False, xyz=(1, 0, 0)), 0.0)
    sim.receive(packet(2, 0.0, xyz=(1, 0, 0)), 0.0)
    assert not sim.enabled
    sim.receive(packet(3, 0.0, enabled=False), 0.0)
    sim.receive(packet(4, 0.0, xyz=(1, 0, 0)), 0.0)
    assert sim.enabled


@pytest.mark.parametrize("axis,sign", [(axis, sign) for axis in range(3) for sign in (-1, 1)])
def test_official_action_moves_physics_in_each_xyz_direction(official, axis, sign):
    sim = GamepadMujoco()
    arm(sim)
    initial = sim.goal_xyz.copy()
    # 官方映射：delta_x=-axis1, delta_y=-axis0, delta_z=-axis3。
    official.axes[(1, 0, 3)[axis]] = -0.3 * sign
    action = official.teleop.get_action()
    for index in range(20):
        now = index * sim.dt
        sim.receive({**packet(index + 2, now), "action": action}, now)
        previous = sim.target.copy()
        sim.advance(now)
        assert np.max(np.abs(sim.target[:5] - previous[:5])) <= JOINT_SPEED * sim.dt + 1e-12
        assert sim.enabled, sim.reason
    assert sign * (sim.goal_xyz[axis] - initial[axis]) > 0.001
    assert np.linalg.norm(sim.data.qpos[sim.qids[:5]] - sim.target[:5]) < 0.1


def test_gripper_and_rb_release_hold_last_target():
    sim = GamepadMujoco()
    arm(sim)
    sim.receive(packet(2, 0, gripper=0), 0)
    sim.advance(0)
    assert sim.target[5] < 1.2
    target = sim.target.copy()
    sim.receive(packet(3, sim.dt, enabled=False, xyz=(1, 0, 0), gripper=0), sim.dt)
    sim.advance(sim.dt)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled


def test_timeout_does_not_automatically_resume():
    sim = GamepadMujoco()
    arm(sim)
    target = sim.target.copy()
    sim.advance(INPUT_TIMEOUT + 0.001)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    sim.receive(packet(2, 1, xyz=(1, 0, 0)), 1)
    assert not sim.enabled
    sim.receive(packet(3, 1, enabled=False), 1)
    sim.receive(packet(4, 1), 1)
    assert sim.enabled


@pytest.mark.parametrize("field,value", [("delta_x", float("nan")), ("delta_z", 2), ("gripper", 3)])
def test_invalid_action_pauses(field, value):
    sim = GamepadMujoco()
    arm(sim)
    message = packet(2, 0)
    message["action"][field] = value
    with pytest.raises(ValueError):
        sim.receive(message, 0)
    assert not sim.enabled


@pytest.mark.parametrize("seq,stamp", [(1, 0), (2, -1), (2, 1)])
def test_duplicate_and_stale_or_future_packets_pause(seq, stamp):
    sim = GamepadMujoco()
    arm(sim)
    with pytest.raises(ValueError):
        sim.receive(packet(seq, stamp), 0)
    assert not sim.enabled


def test_unreachable_ik_is_rejected_without_partial_target(monkeypatch):
    sim = GamepadMujoco()
    arm(sim)
    target = sim.target.copy()
    monkeypatch.setattr(sim.ik, "solve", lambda *args, **kwargs: (np.zeros(5), 0.05))
    sim.receive(packet(2, 0, xyz=(1, 0, 0), gripper=0), 0)
    sim.advance(0)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    assert sim.reason.startswith("IK 未收敛（dls-grasp）")


def test_collision_preview_rejects_target(monkeypatch):
    sim = GamepadMujoco()
    arm(sim)
    target = sim.target.copy()
    monkeypatch.setattr(sim, "_collision", lambda data: data is sim.preview)
    sim.receive(packet(2, 0, gripper=0), 0)
    sim.advance(0)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    assert sim.reason == "目标姿态碰撞"


def test_workspace_boundary_rejects_motion():
    sim = GamepadMujoco()
    arm(sim)
    target = sim.target.copy()
    sim.goal_xyz[0] = 0.34
    sim.receive(packet(2, 0, xyz=(1, 0, 0)), 0)
    sim.advance(0)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    assert sim.reason == "超出末端工作区"


def test_real_table_collision_is_detected():
    import mujoco

    sim = GamepadMujoco()
    # 将桌面抬到腕部，使用真实接触计算验证碰撞分类，而非只替换判定函数。
    table = sim.model.geom("table").id
    sim.model.geom_pos[table, 2] = sim.data.xpos[sim.gripper_id, 2]
    sim.model.geom_size[table, 2] = 0.10
    mujoco.mj_forward(sim.model, sim.data)
    assert sim._collision(sim.data)


def test_packet_json_roundtrip_accepts_official_numpy_values(official):
    action = official.teleop.get_action()
    message = {**packet(0, 0, enabled=False), "action": {key: float(value) for key, value in action.items()}}
    sim = GamepadMujoco()
    sim.receive(json.loads(json.dumps(message, allow_nan=False)), 0)
    assert sim.can_enable


@pytest.fixture
def entry():
    return load_script("scripts/teleoperate_gamepad.py", "gamepad_live_entry")


@pytest.mark.parametrize(
    "raw,delta,gripper,intervention",
    [
        ([128, 127, 128, 127, 8, 0, 0, 255], [0, 0, 0], "stay", False),
        ([128, 0, 128, 127, 8, 0, 0, 255], [1, 0, 0], "stay", False),
        ([0, 127, 128, 127, 8, 0, 0, 255], [0, 1, 0], "stay", False),
        ([128, 127, 128, 0, 8, 0, 0, 255], [0, 0, 1], "stay", False),
        ([128, 127, 128, 127, 8, 2, 0, 255], [0, 0, 0], "stay", True),
        ([128, 127, 128, 127, 8, 6, 0, 255], [0, 0, 0], "close", True),
        ([128, 127, 128, 127, 8, 10, 0, 255], [0, 0, 0], "open", True),
    ],
)
def test_f310_report_layout_uses_official_hid_parser(entry, raw, delta, gripper, intervention):
    # F310 真机报文没有 report ID；补齐后继续使用原版官方解析器。
    device = SimpleNamespace(read=lambda size: raw, close=lambda: None)
    pad = gamepad_utils.GamepadControllerHID()
    pad.device = entry.CheckedHIDReader(device)
    pad._update()
    np.testing.assert_allclose(pad.get_deltas(), delta)
    assert pad.gripper_command() == gripper
    assert pad.should_intervene() == intervention


def test_hid_disconnect_is_not_swallowed(entry):
    def disconnected(size):
        raise OSError("unplugged")

    pad = gamepad_utils.GamepadControllerHID()
    pad.device = entry.CheckedHIDReader(SimpleNamespace(read=disconnected))
    with pytest.raises(RuntimeError, match="HID 连接断开"):
        pad._update()


def test_hid_empty_poll_and_invalid_report_length(entry):
    reader = entry.CheckedHIDReader(SimpleNamespace(read=lambda size: []))
    assert reader.read(64) == []
    reader.device = SimpleNamespace(read=lambda size: [0] * 7)
    with pytest.raises(RuntimeError, match="报文长度异常"):
        reader.read(64)


@pytest.mark.parametrize("product_id,accepted", [(0xC216, True), (0xC542, False)])
def test_macos_entry_selects_only_f310_hid(entry, monkeypatch, product_id, accepted):
    import lerobot.teleoperators.gamepad as official_module

    closed = []
    config_used = []
    device = SimpleNamespace(read=lambda size: [128, 127, 128, 127, 8, 0, 0, 255])

    def make_teleop(config):
        config_used.append(config)
        return SimpleNamespace(
            hidapi_fallback=config.hidapi_fallback,
            gamepad=SimpleNamespace(
                running=True,
                device=device,
                device_info={"vendor_id": 0x046D, "product_id": product_id, "product_string": "test"},
            ),
            connect=lambda: None,
            disconnect=lambda: closed.append(True),
        )

    monkeypatch.setattr(entry.sys, "platform", "darwin")
    monkeypatch.setattr(official_module, "GamepadTeleop", make_teleop)
    if accepted:
        teleop = entry.connect_gamepad()
        assert isinstance(teleop.gamepad.device, entry.CheckedHIDReader)
        assert not closed
    else:
        with pytest.raises(RuntimeError, match="D 模式"):
            entry.connect_gamepad()
        assert closed
    assert config_used[0].hidapi_fallback
