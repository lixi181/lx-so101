"""SO-101 ACT 交互式闭环推理（本机，无 GPU 需求）——官方 select_action 语义版。

功能
----
- 加载已训练的 ACT checkpoint（AutoDL 训练产物，已拷回本机）。
- 打开 MuJoCo Viewer 和独立的 Cube 控制面板；摆放模式通过面板滑块设置
  cube 的 x/y/z，点击“开始任务”后才执行抓取-放置任务。
- 采用官方 `policy.select_action()` 的 action queue 机制：每推理一次生成
  100 步动作 chunk 入队，逐帧连贯执行完才重新推理（避免动作 chunk 被反复
  截断替换导致抓取失败/卡住）。
- cube 为 freejoint；控制面板提供 x/y/z 三个横向滑块，以及开始和重置按钮。

用法
----
在项目根目录执行：

    ./.venv/bin/mjpython scripts/interactive_infer.py
    ./.venv/bin/python scripts/interactive_infer.py --no-viewer --max-ticks 60   # 无头冒烟

可选参数
--------
--checkpoint PATH   模型目录（默认 outputs/act_so101_pickplace_v3/checkpoints/last/pretrained_model）
--device DEVICE     mps（默认，Apple GPU）/ cpu
--action-steps N    每次推理后连贯执行的步数（官方 n_action_steps，默认 100；
                    调小如 20/50 可缩短动作重规划间隔，动作仍连贯）
--fps N             控制节流频率（默认 30，与训练数据一致）
--no-viewer         无头冒烟模式：不开窗口，不创建图形上下文，打印动作后自动退出
--max-ticks N       无头模式下最多执行 N 个控制 tick
--max-attempts N    抓取失败后的最大尝试次数（默认 3；0 表示无限重试）
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from convert_to_lerobot import GRIPPER_SCALE_MAX, gripper_limits, to_lerobot_units
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from mujoco import viewer

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/act_so101_pickplace_v3/checkpoints/last/pretrained_model"
)
TASK = "Pick up the cube and place it in the box"
ROBOT_TYPE = "so101_follower"
FRONT_CAM, WRIST_CAM = "front_camera", "wrist_camera"
# 场景相机名 → 模型输入特征键（训练时数据集键：observation.images.front / .wrist）
CAM_TO_FEATURE = {FRONT_CAM: "front", WRIST_CAM: "wrist"}
IMG_H, IMG_W = 240, 320
PHYSICS_PER_TICK = 16  # 480Hz 物理步 → 每控制 tick 1/30s 仿真时间
GRASP_LIFT_THRESHOLD_M = 0.070
GRASP_CHECK_TICKS = 320

def sim_to_policy(qpos6: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """仿真 6 维 qpos（弧度 + 夹爪关节弧度）→ 策略单位（前 5 度、夹爪 0-100）。"""
    return to_lerobot_units(np.asarray(qpos6, dtype=np.float64), limits).astype(np.float32)


def policy_to_sim(action6: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """策略输出 action（前 5 度、夹爪 0-100）→ 仿真关节目标（弧度）。"""
    a = np.asarray(action6, dtype=np.float64)
    out = np.empty_like(a)
    out[:5] = np.deg2rad(a[:5])
    low, high = limits
    out[5] = low + np.clip(a[5], 0.0, GRIPPER_SCALE_MAX) / GRIPPER_SCALE_MAX * (high - low)
    return out


def blank_obs() -> dict:
    """返回无头冒烟用的占位图像，不触发 OpenGL/CoreGraphics 初始化。"""
    frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    return {
        "observation.images.front": frame.copy(),
        "observation.images.wrist": frame.copy(),
    }


def render_obs(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """按训练一致的相机/尺寸渲染两张图，返回 {observation.images.<front|wrist>: (H,W,C) uint8}。"""
    frames = {}
    for cam in (FRONT_CAM, WRIST_CAM):
        renderer.update_scene(data, camera=cam)
        frames[f"observation.images.{CAM_TO_FEATURE[cam]}"] = renderer.render().copy()
    return frames


def start_control_panel(initial_xyz: np.ndarray) -> dict:
    """启动独立普通 Python 控制面板，避开 mjpython 的 macOS Cocoa 运行时。"""
    socket_path = os.path.join(tempfile.gettempdir(), f"so101_cube_panel_{os.getpid()}.sock")
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.setblocking(False)

    python_executable = Path(sys.executable)
    if python_executable.name == "mjpython":
        sibling = python_executable.with_name("python")
        if sibling.exists():
            python_executable = sibling
    panel_script = Path(__file__).with_name("cube_control_panel.py")
    log_path = f"{socket_path}.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                str(python_executable),
                str(panel_script),
                socket_path,
                "--x",
                f"{float(initial_xyz[0]):.6f}",
                "--y",
                f"{float(initial_xyz[1]):.6f}",
                "--z",
                f"{float(initial_xyz[2]):.6f}",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    state = {
        "x": float(initial_xyz[0]),
        "y": float(initial_xyz[1]),
        "z": float(initial_xyz[2]),
        "start": False,
        "reset": False,
        "closed": False,
    }
    return {
        "server": server,
        "client": None,
        "process": process,
        "socket_path": socket_path,
        "log_path": log_path,
        "buffer": "",
        "state": state,
    }


def send_control_panel(panel: dict, message: dict) -> None:
    client = panel["client"]
    if client is None or panel["state"]["closed"]:
        return
    try:
        client.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())
    except OSError:
        panel["state"]["closed"] = True


def poll_control_panel(panel: dict) -> list[dict]:
    """非阻塞接收面板命令。"""
    if panel["state"]["closed"]:
        return []
    if panel["client"] is None:
        process = panel.get("process")
        if process is not None and process.poll() is not None:
            log_path = panel.get("log_path")
            details = ""
            if log_path:
                try:
                    details = Path(log_path).read_text(encoding="utf-8").strip()
                except OSError:
                    pass
            print(f"[interactive] Cube 控制面板已退出（returncode={process.returncode}）")
            if details:
                print(f"[interactive] 控制面板日志: {details}")
            panel["state"]["closed"] = True
            return []
        ready, _, _ = select.select([panel["server"]], [], [], 0)
        if ready:
            client, _ = panel["server"].accept()
            client.setblocking(False)
            panel["client"] = client
    client = panel["client"]
    if client is None:
        return []
    ready, _, _ = select.select([client], [], [], 0)
    if not ready:
        return []
    try:
        chunk = client.recv(65536)
    except BlockingIOError:
        return []
    except OSError:
        panel["state"]["closed"] = True
        return []
    if not chunk:
        panel["state"]["closed"] = True
        return []
    panel["buffer"] += chunk.decode()
    lines = panel["buffer"].split("\n")
    panel["buffer"] = lines.pop()
    return [json.loads(line) for line in lines if line]


def close_control_panel(panel: dict) -> None:
    panel["state"]["closed"] = True
    send_control_panel(panel, {"type": "close"})
    for key in ("client", "server"):
        sock = panel.get(key)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
    process = panel.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        os.unlink(panel["socket_path"])
    except FileNotFoundError:
        pass
    try:
        os.unlink(panel["log_path"])
    except FileNotFoundError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 ACT 交互式闭环推理（select_action 语义）")
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument(
        "--device", choices=("mps", "cpu"), default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    ap.add_argument("--action-steps", type=int, default=100, help="每次推理后连贯执行步数 (1-100)")
    ap.add_argument("--fps", type=int, default=30, help="控制节流频率")
    ap.add_argument("--no-viewer", action="store_true", help="无头冒烟模式，不打开窗口或创建图形上下文")
    ap.add_argument("--max-ticks", type=int, default=60, help="无头模式控制 tick 数")
    ap.add_argument("--max-attempts", type=int, default=3, help="抓取失败后的最大尝试次数，0 表示无限重试")
    args = ap.parse_args()
    action_steps = max(1, min(args.action_steps, 100))
    max_attempts = max(0, args.max_attempts)
    dt = 1.0 / max(1, args.fps)

    print(f"[infer] checkpoint: {args.checkpoint}")
    print(f"[infer] device: {args.device}")
    print(f"[infer] task: {TASK}")
    print(f"[infer] action_steps: {action_steps}（每次推理后连贯执行的步数）")

    # ---- 场景 ----
    model, data = load_model(FIXED_SCENE_PATH)
    ready_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
    if ready_id < 0:
        raise RuntimeError("场景中找不到 keyframe 'folded_ready'")
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_joint_id < 0 or cube_body_id < 0:
        raise RuntimeError("场景中找不到 cube freejoint/body")
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])
    mujoco.mj_resetDataKeyframe(model, data, ready_id)
    mujoco.mj_forward(model, data)
    limits = gripper_limits(FIXED_SCENE_PATH)
    print(f"[infer] gripper limits: {limits[0]:.4f} ~ {limits[1]:.4f} rad")

    # ---- 模型 ----
    t0 = time.perf_counter()
    policy = ACTPolicy.from_pretrained(args.checkpoint, device=args.device)
    policy.to(args.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )
    # 覆盖 queue 长度（官方 n_action_steps 语义），并重建 action queue
    policy.config.n_action_steps = action_steps
    policy.reset()
    print(f"[infer] policy+processors loaded in {time.perf_counter() - t0:.1f}s, "
          f"params={sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

    # `mujoco.Renderer` 在 macOS 通过 CoreGraphics 创建 CGL 上下文。无头冒烟只验证
    # 策略和物理循环；在没有 WindowServer 的环境里直接初始化它会失败或卡住。
    renderer = None if args.no_viewer else mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    def reset_to_ready() -> None:
        """恢复 folded_ready，并让下一次任务从摆放模式开始。"""
        nonlocal obs_cache, phase, attempt_number, attempt_tick, grasp_checked, attempt_cube_qpos
        mujoco.mj_resetDataKeyframe(model, data, ready_id)
        mujoco.mj_forward(model, data)
        policy.reset()
        obs_cache = None
        phase = "placing"
        attempt_number = 0
        attempt_tick = 0
        grasp_checked = False
        attempt_cube_qpos = None
        print("[interactive] 已重置，请重新摆放 cube，按 Enter 开始", flush=True)

    def consume_panel_cmd(window, panel: dict) -> None:
        """在主线程消费面板按钮事件并更新交互状态。"""
        state = panel["state"]
        nonlocal phase, obs_cache, t_run, attempt_number, attempt_tick
        nonlocal grasp_checked, attempt_cube_qpos, attempt_cube_start_z
        if state["reset"]:
            state["reset"] = False
            state["start"] = False
            with window.lock():
                reset_to_ready()
            send_control_panel(panel, {"type": "pose", "x": data.qpos[cube_qpos_adr],
                                       "y": data.qpos[cube_qpos_adr + 1],
                                       "z": data.qpos[cube_qpos_adr + 2]})
            send_control_panel(panel, {"type": "phase", "phase": "placing"})
            return
        if state["start"] and phase == "placing":
            state["start"] = False
            attempt_cube_qpos = data.qpos[cube_qpos_adr : cube_qpos_adr + 7].copy()
            attempt_cube_start_z = float(data.xpos[cube_body_id, 2])
            attempt_number = 1
            attempt_tick = 0
            grasp_checked = False
            policy.reset()
            obs_cache = None
            phase = "running"
            t_run = time.perf_counter()
            send_control_panel(panel, {"type": "phase", "phase": "running"})
            print("[interactive] 任务开始", flush=True)

    def apply_panel_pose(window, panel: dict) -> None:
        """在摆放阶段把面板的 x/y/z 写回 cube freejoint。"""
        if phase != "placing" or panel["state"]["closed"]:
            return
        requested = np.array(
            [panel["state"]["x"], panel["state"]["y"], panel["state"]["z"]],
            dtype=np.float64,
        )
        with window.lock():
            current = data.qpos[cube_qpos_adr : cube_qpos_adr + 3]
            if not np.allclose(current, requested, atol=1e-12):
                data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = requested
                data.qvel[cube_dof_adr : cube_dof_adr + 6] = 0.0
                mujoco.mj_forward(model, data)

    def cube_contact_names() -> set[str]:
        """返回当前与 cube 接触的 body 名称。"""
        names = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if cube_body_id not in (body1, body2):
                continue
            other = body2 if body1 == cube_body_id else body1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
            names.add(name or str(other))
        return names

    def restore_attempt_state() -> None:
        """恢复机械臂 folded_ready，并保留 Enter 时的 cube 位姿。"""
        if attempt_cube_qpos is None:
            return
        mujoco.mj_resetDataKeyframe(model, data, ready_id)
        data.qpos[cube_qpos_adr : cube_qpos_adr + 7] = attempt_cube_qpos
        data.qvel[cube_dof_adr : cube_dof_adr + 6] = 0.0
        data.ctrl[:6] = data.qpos[:6]
        mujoco.mj_forward(model, data)

    def check_grasp_and_retry() -> None:
        """在抬升阶段后判定抓取，失败则恢复并重新开始一次尝试。"""
        nonlocal phase, obs_cache, attempt_number, attempt_tick, grasp_checked
        if phase != "running" or grasp_checked or attempt_tick < GRASP_CHECK_TICKS:
            return
        lifted = float(data.xpos[cube_body_id, 2] - attempt_cube_start_z)
        contacts = cube_contact_names()
        bilateral_contact = {"gripper", "moving_jaw_so101_v1"}.issubset(contacts)
        grasp_ok = lifted >= GRASP_LIFT_THRESHOLD_M and bilateral_contact and "world" not in contacts
        if grasp_ok:
            grasp_checked = True
            print(
                f"[interactive] 抓取成功：抬升 {lifted:.3f} m，接触 {sorted(contacts)}，继续执行放置",
                flush=True,
            )
            return

        print(
            f"[interactive] 第 {attempt_number} 次抓取失败：抬升 {lifted:.3f} m，"
            f"接触 {sorted(contacts) or '无'}",
            flush=True,
        )
        if max_attempts and attempt_number >= max_attempts:
            restore_attempt_state()
            policy.reset()
            obs_cache = None
            phase = "placing"
            send_control_panel(panel, {"type": "phase", "phase": "placing"})
            print(
                f"[interactive] 已达到最大尝试次数 {max_attempts}，"
                "请重新摆放 cube 后按 Enter，或按 F8 重置",
                flush=True,
            )
            return

        restore_attempt_state()
        policy.reset()
        obs_cache = None
        attempt_number += 1
        attempt_tick = 0
        grasp_checked = False
        print(f"[interactive] 自动开始第 {attempt_number} 次抓取尝试", flush=True)

    # ---- 推理循环（官方 select_action：action queue 连贯执行）----
    obs_cache = None
    phase = "placing"
    tick = 0
    t_run = time.perf_counter()
    attempt_number = 0
    attempt_tick = 0
    grasp_checked = False
    attempt_cube_qpos: np.ndarray | None = None
    attempt_cube_start_z = 0.0

    def run_one_tick(window=None) -> np.ndarray:
        """执行一个控制 tick，返回本次下发的关节目标（弧度）。"""
        nonlocal obs_cache, tick, attempt_tick
        # action queue 为空时才渲染+推理（queue 非空时 select_action 只弹队列，不需新观测）
        if len(policy._action_queue) == 0:
            image_obs = blank_obs() if renderer is None else render_obs(renderer, model, data)
            obs = {
                **image_obs,
                "observation.state": sim_to_policy(data.qpos[:6], limits),
            }
            obs = prepare_observation_for_inference(obs, torch.device(args.device), TASK, ROBOT_TYPE)
            obs = preprocessor(obs)
            obs_cache = obs
        with torch.inference_mode():
            action = postprocessor(policy.select_action(obs_cache))
        target = policy_to_sim(action[0].cpu().numpy(), limits)
        data.ctrl[:6] = target
        for _ in range(PHYSICS_PER_TICK):
            mujoco.mj_step(model, data)
        tick += 1
        attempt_tick += 1
        return target

    if args.no_viewer:
        print("[infer] no-viewer 冒烟模式开始（select_action / action queue；占位图像，不初始化 Renderer）")
        for _ in range(args.max_ticks):
            target = run_one_tick()
            if tick % 10 == 0:
                print(f"[infer] tick {tick} queue={len(policy._action_queue):3d} "
                      f"ctrl={np.round(target, 3)}", flush=True)
        print(f"[infer] 无头冒烟完成: {tick} ticks, "
              f"qpos[:6]={np.round(data.qpos[:6], 3)}, 用时 {time.perf_counter() - t_run:.1f}s")
        return

    # ---- 交互模式 ----
    panel = start_control_panel(data.qpos[cube_qpos_adr : cube_qpos_adr + 3])
    print("[interactive] 已打开 Cube 控制面板：调整 x/y/z 后点击“开始任务”；"
          "点击“重置并重新摆放”可重新设置位置")
    try:
        with viewer.launch_passive(model, data) as window:
            while window.is_running() and not panel["state"]["closed"]:
                t0 = time.perf_counter()
                for command in poll_control_panel(panel):
                    command_type = command.get("type")
                    if command_type == "pose":
                        panel["state"].update(
                            x=float(command["x"]),
                            y=float(command["y"]),
                            z=float(command["z"]),
                        )
                    elif command_type == "start":
                        panel["state"]["start"] = True
                    elif command_type == "reset":
                        panel["state"]["reset"] = True
                    elif command_type == "close":
                        panel["state"]["closed"] = True
                consume_panel_cmd(window, panel)
                apply_panel_pose(window, panel)
                if phase == "running":
                    with window.lock():
                        run_one_tick(window)
                        check_grasp_and_retry()
                    window.sync()
                else:
                    window.sync()
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        close_control_panel(panel)
    print(f"[interactive] viewer 已退出，共执行 {tick} 个控制 tick（{(time.perf_counter() - t_run):.1f}s）")


if __name__ == "__main__":
    main()
