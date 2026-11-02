"""IK 启动选项、TCP 一致性和无窗口物理回归。"""

from __future__ import annotations

import contextlib
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from module_loader import load_script

from so_ws.teleoperators.gamepad_ik import DEFAULT_IK, IK_MODES, LEGACY_IK_MODES
from so_ws.teleoperators.gamepad_mujoco import INPUT_TIMEOUT, JOINT_SPEED, GamepadMujoco


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    import hid
    import mujoco.viewer
    import pygame
    import serial

    def forbidden(*args, **kwargs):
        raise AssertionError("模式测试禁止初始化硬件或 GUI")

    for obj, name in [
        (hid, "device"),
        (serial, "Serial"),
        (pygame, "init"),
        (pygame.joystick, "init"),
        (pygame.display, "init"),
        (mujoco.viewer, "launch_passive"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)


@pytest.fixture
def entry():
    return load_script("scripts/teleoperate_gamepad.py", "gamepad_ik_entry")


def packet(seq, now, enabled=True, xyz=(0, 0, 0)):
    return {
        "seq": seq,
        "stamp": now,
        "enabled": enabled,
        "action": {"delta_x": xyz[0], "delta_y": xyz[1], "delta_z": xyz[2], "gripper": 1},
    }


@pytest.mark.parametrize(
    "mode,position_reachable",
    [
        ("placo-position", True),
        ("placo-soft", False),
        ("dls-position", True),
        ("dls-grasp", False),
    ],
)
def test_modes_have_distinct_reachability_without_mutating_physics(mode, position_reachable):
    sim = GamepadMujoco(ik_mode=mode)
    before = sim.data.qpos.copy()
    q, error = sim.ik.solve(np.array([0.22, 0, 0.13]), sim.target[:5])
    assert (error <= 0.002) == position_reachable
    assert np.all(q >= sim.ranges[:5, 0] - 1e-7)
    assert np.all(q <= sim.ranges[:5, 1] + 1e-7)
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert sim.data.time == 0


@pytest.mark.parametrize("mode", ["placo-position", "dls-position"])
def test_position_modes_can_raise_above_old_grasp_limit(mode):
    sim = GamepadMujoco(ik_mode=mode)
    sim.receive(packet(0, 0, enabled=False), 0)
    for index in range(100):
        now = index * sim.dt
        sim.receive(packet(index + 1, now, xyz=(0, 0, 1)), now)
        previous = sim.target.copy()
        sim.advance(now)
        assert sim.enabled, sim.reason
        assert np.max(np.abs(sim.target[:5] - previous[:5])) <= JOINT_SPEED * sim.dt + 1e-12
    assert sim.goal_xyz[2] > 0.16
    assert np.isfinite(sim.data.qpos).all()


@pytest.mark.parametrize("mode", LEGACY_IK_MODES)
def test_all_modes_require_rearm_after_timeout(mode):
    sim = GamepadMujoco(ik_mode=mode)
    sim.receive(packet(0, 0, enabled=False), 0)
    sim.receive(packet(1, 0), 0)
    assert sim.enabled
    target = sim.target.copy()
    sim.advance(INPUT_TIMEOUT + 0.01)
    np.testing.assert_array_equal(sim.target, target)
    assert not sim.enabled
    sim.receive(packet(2, 1), 1)
    assert not sim.enabled


def test_placo_tcp_matches_mujoco_over_multiple_joint_poses():
    sim = GamepadMujoco(ik_mode="placo-position")
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = rng.uniform(sim.ranges[:5, 0] + 0.02, sim.ranges[:5, 1] - 0.02)
        official = sim.ik.kinematics.forward_kinematics(np.rad2deg(q))
        local = sim.ik._pose(q)
        assert np.linalg.norm(official[:3, 3] - local[:3, 3]) < 0.00002


@pytest.mark.parametrize("mode", [*IK_MODES, None])
def test_cli_dispatches_selected_or_default_ik(entry, monkeypatch, mode):
    selected = []
    args = ["teleoperate_gamepad.py"] + ([] if mode is None else ["--ik", mode])
    monkeypatch.setattr(entry.sys, "argv", args)
    monkeypatch.setattr(entry, "run_gamepad", lambda value: selected.append(value))
    assert entry.main() == 0
    assert selected == [DEFAULT_IK if mode is None else mode]


@pytest.mark.parametrize("mode", ["invalid", "placo-position", "placo-soft"])
def test_invalid_cli_mode_never_opens_gamepad(entry, monkeypatch, mode):
    monkeypatch.setattr(entry.sys, "argv", ["teleoperate_gamepad.py", "--ik", mode])
    monkeypatch.setattr(entry, "run_gamepad", lambda mode: pytest.fail("非法策略不应启动"))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", IK_MODES)
def test_frontend_forwards_mode_to_worker(entry, monkeypatch, mode):
    import pygame

    called = []
    read_fd, write_fd = os.pipe()
    os.write(write_fd, (json.dumps({"ready": True, "dt": 1 / 30, "ik": mode}) + "\n").encode())
    os.close(write_fd)
    stdout = os.fdopen(read_fd, "rb")
    worker = SimpleNamespace(stdin=io.StringIO(), stdout=stdout, poll=lambda: 0, wait=lambda timeout: 0)
    teleop = SimpleNamespace(gamepad=None, disconnect=lambda: None)
    monkeypatch.setattr(entry, "connect_gamepad", lambda: teleop)
    monkeypatch.setattr(entry.subprocess, "Popen", lambda command, **kwargs: called.append(command) or worker)
    monkeypatch.setattr(pygame.display, "init", lambda: None)
    monkeypatch.setattr(pygame.display, "set_mode", lambda size: None)
    monkeypatch.setattr(pygame.display, "set_caption", lambda caption: None)
    monkeypatch.setattr(pygame.font, "init", lambda: None)
    monkeypatch.setattr(pygame.font, "Font", lambda *args: None)
    entry.run_gamepad(mode)
    assert called[0][-3:] == ["--worker", "--ik", mode]


@pytest.mark.parametrize("mode", IK_MODES)
def test_worker_reports_actual_selected_mode(entry, monkeypatch, capsys, mode):
    import mujoco.viewer

    viewer = SimpleNamespace(cam=SimpleNamespace(lookat=np.zeros(3), distance=0), is_running=lambda: False)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *args: contextlib.nullcontext(viewer))
    entry.run_worker(mode)
    message = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert message["ik"] == mode
    assert message["ready"]


def test_ready_protocol_ignores_native_library_diagnostics(entry, capsys):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'Placo native diagnostic\n{"ready": true, "ik": "placo-position", "dt": 0.0333}\n')
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stdout:
        message = entry.wait_worker_ready(SimpleNamespace(stdout=stdout))
    assert message["ik"] == "placo-position"
    assert "Placo native diagnostic" in capsys.readouterr().err


def test_ready_protocol_rejects_worker_early_exit(entry):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stdout, pytest.raises(RuntimeError, match="就绪前退出"):
        entry.wait_worker_ready(SimpleNamespace(stdout=stdout))
