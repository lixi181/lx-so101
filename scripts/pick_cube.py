"""在 MuJoCo 中执行一次抓取、放置到盒子并返回初始姿态的动作。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.ik_controller import arm_addresses, grasp_point, gripper_yaw, solve_ik
from controllers.motion_profiles import MotionProfile, get_motion_profile
from controllers.square_grasp_planner import (
    minimum_jerk,
    select_square_grasp_yaw,
    square_grasp_yaw_candidates,
    wrap_to_pi,
)
from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model, scene_timestep
from task_sampling import SCENE_BOX_XY, SamplerConfig, TaskSample, apply_task, derive_seed, sample_task

READY_KEYFRAME = "folded_ready"
GRIPPER_BODY = "gripper"
CUBE_BODY = "cube"
TARGET_BOX_BODY = "target_box"
GRIPPER_OPEN = 1.20
GRIPPER_CLOSED = -0.17
CLEARANCE_Q = np.array([-0.0187914029, -1.5800000000, 1.5000000000, 1.1500000000, 0.0])
ACTIVE_PROFILE = get_motion_profile("simulation_fast")
PHYSICS_PER_TICK = ACTIVE_PROFILE.physics_per_tick
# 模块级默认值；run() 每次都会按实际场景的 timestep 重新调用 configure_profile。
# 这里从场景读取步长而不是写死常量，避免改了 XML 之后默认值失配。
CONTROL_PERIOD = PHYSICS_PER_TICK * scene_timestep()
CARTESIAN_TICKS = ACTIVE_PROFILE.cartesian_ticks
# 观测回调：observer(model, data, phase, action)。
# 回调在动作被执行之前调用，因此 data 是 s_t，action 是将要在第 t 个控制 tick
# 内插值执行到末尾的绝对关节位置目标 a_t；两者严格同拍（obs[t] 对应 action[t]）。
Observer = Callable[[mujoco.MjModel, mujoco.MjData, str, np.ndarray], None]
# 最近一次进入的控制阶段。run() 用它给失败结果标注阶段，采集器在没有结构化结果时
# 用它兜底；脚本是单线程的，和 ACTIVE_PROFILE 一样是模块级状态。
CURRENT_PHASE = "initial"


def note_phase(phase: str) -> None:
    global CURRENT_PHASE
    CURRENT_PHASE = phase


@dataclass
class RunResult:
    """一次专家演示的结构化结果。"""

    success: bool
    grasp_ok: bool
    placed: bool
    returned: bool
    failure_phase: str | None
    failure_reason: str | None

    def __bool__(self) -> bool:
        return self.success


def configure_profile(profile: MotionProfile, timestep: float) -> None:
    global ACTIVE_PROFILE, PHYSICS_PER_TICK, CONTROL_PERIOD, CARTESIAN_TICKS
    ACTIVE_PROFILE = profile
    PHYSICS_PER_TICK = profile.physics_per_tick
    CONTROL_PERIOD = profile.physics_per_tick * timestep
    CARTESIAN_TICKS = profile.cartesian_ticks


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError(f"找不到 body: {name}")
    return result


def sync(window: object | None) -> None:
    if window is not None:
        window.sync()


def reset_to_ready(model: mujoco.MjModel, data: mujoco.MjData, task: TaskSample | None = None) -> None:
    """复位到 folded_ready，写入任务初始状态，并把 ctrl 预置为第一帧的命令。

    采集和开环回放共用这一个入口，保证回放从与采集完全相同的初始状态出发。
    """
    ready_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, READY_KEYFRAME)
    if ready_id < 0:
        raise ValueError(f"找不到 keyframe: {READY_KEYFRAME}")
    mujoco.mj_resetDataKeyframe(model, data, ready_id)
    if task is not None:
        apply_task(model, data, task)
    mujoco.mj_forward(model, data)
    data.ctrl[:5] = data.qpos[:5]
    data.ctrl[5] = GRIPPER_OPEN


def step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: object | None,
    count: int,
    phase: str = "hold",
    observer: Observer | None = None,
) -> None:
    """保持当前 ctrl 推进 count 个物理步。

    count 必须是控制 tick 的整数倍，否则采样节奏会出现不完整的 tick；每个 tick
    开始时先记录 (s_t, a_t)，再执行该 tick 的物理步。
    """
    assert count % PHYSICS_PER_TICK == 0, f"{phase}: 物理步数 {count} 必须是 tick 的整数倍"
    note_phase(phase)
    for _ in range(count // PHYSICS_PER_TICK):
        if window is not None and not window.is_running():
            raise InterruptedError("用户关闭了抓取窗口")
        if observer is not None:
            observer(model, data, phase, data.ctrl.copy())
        for _ in range(PHYSICS_PER_TICK):
            mujoco.mj_step(model, data)
        if window is not None:
            sync(window)
            time.sleep(CONTROL_PERIOD)


def advance_to_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: object | None,
    target_ctrl: np.ndarray,
    physics_steps: int | None = None,
    phase: str = "control",
    observer: Observer | None = None,
) -> None:
    """在一个控制 tick 内插值到新 setpoint，避免 position ctrl 阶跃。

    限速之后的 target_ctrl 才是真正会被执行的命令，因此交给 observer 的是限速后的值。
    记录发生在插值和步进之前：obs[t] 是执行 action[t] 之前的状态。
    """
    if physics_steps is None:
        physics_steps = PHYSICS_PER_TICK
    note_phase(phase)
    assert physics_steps % PHYSICS_PER_TICK == 0, f"{phase}: 物理步数 {physics_steps} 必须是 tick 的整数倍"
    if observer is not None:
        assert physics_steps == PHYSICS_PER_TICK, (
            f"{phase}: 带 observer 的 advance_to_ctrl 必须恰好推进一个控制 tick，"
            f"当前 {physics_steps} != {PHYSICS_PER_TICK}"
        )
    start_ctrl = data.ctrl.copy()
    target_ctrl = np.asarray(target_ctrl, dtype=np.float64)
    max_delta = np.full_like(target_ctrl, np.inf)
    max_delta[:5] = ACTIVE_PROFILE.max_joint_velocity_rad_s * CONTROL_PERIOD
    max_delta[5] = ACTIVE_PROFILE.max_gripper_velocity_rad_s * CONTROL_PERIOD
    target_ctrl = start_ctrl + np.clip(target_ctrl - start_ctrl, -max_delta, max_delta)
    if observer is not None:
        observer(model, data, phase, target_ctrl.copy())
    for index in range(physics_steps):
        if window is not None and not window.is_running():
            raise InterruptedError("用户关闭了抓取窗口")
        fraction = (index + 1) / physics_steps
        data.ctrl[:] = start_ctrl * (1.0 - fraction) + target_ctrl * fraction
        mujoco.mj_step(model, data)
    data.ctrl[:] = target_ctrl
    if window is not None:
        sync(window)
        time.sleep(CONTROL_PERIOD)


def move_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: object | None,
    target: np.ndarray,
    qpos_ids: list[int],
    dof_ids: list[int],
    gripper_id: int,
    label: str,
    observer: Observer | None = None,
    target_yaw: float | None = None,
) -> None:
    start = grasp_point(model, data, gripper_id).copy()
    for fraction in np.linspace(0.0, 1.0, CARTESIAN_TICKS)[1:]:
        progress = minimum_jerk(fraction)
        waypoint = start * (1.0 - progress) + target * progress
        q, error = solve_ik(model, data, waypoint, qpos_ids, dof_ids, gripper_id, target_yaw=target_yaw)
        target_ctrl = data.ctrl.copy()
        target_ctrl[:5] = q
        advance_to_ctrl(model, data, window, target_ctrl, phase=label, observer=observer)
    final_error = np.linalg.norm(target - grasp_point(model, data, gripper_id))
    yaw_error = 0.0
    if target_yaw is not None:
        yaw_error = abs(wrap_to_pi(target_yaw - gripper_yaw(model, data, gripper_id)))
        print(f"{label}: 末端误差 {final_error:.4f} m, yaw 误差 {yaw_error:.4f} rad")
    else:
        print(f"{label}: 末端误差 {final_error:.4f} m")
    if error > 0.03 or final_error > 0.03:
        raise RuntimeError(f"{label} 的 IK 未收敛，误差 {error:.4f}")
    if yaw_error > 0.05:
        raise RuntimeError(f"{label} 的 yaw 未收敛，误差 {yaw_error:.4f} rad")


def return_to_ready(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: object | None,
    ready_arm: np.ndarray,
    ready_gripper: float,
    profile: MotionProfile,
    observer: Observer | None = None,
) -> None:
    """在 clearance 连续通过；同方向关节共享边界速度，反向关节自然换向。"""
    points = (data.qpos[:5].copy(), CLEARANCE_Q, ready_arm)
    ticks = (profile.clearance_ticks - 1, profile.folded_ready_ticks - 1)
    durations = np.asarray(ticks) * CONTROL_PERIOD
    slopes = ((points[1] - points[0]) / durations[0], (points[2] - points[1]) / durations[1])
    tangent = np.zeros(5)
    same_direction = slopes[0] * slopes[1] > 0
    tangent[same_direction] = (
        2
        * slopes[0][same_direction]
        * slopes[1][same_direction]
        / (slopes[0][same_direction] + slopes[1][same_direction])
    )
    velocities = (np.zeros(5), tangent, np.zeros(5))
    for segment, label in enumerate(("回程 clearance", "回程 folded_ready")):
        for tick in range(1, ticks[segment] + 1):
            u = tick / ticks[segment]
            target_ctrl = data.ctrl.copy()
            target_ctrl[:5] = (
                (2 * u**3 - 3 * u**2 + 1) * points[segment]
                + (u**3 - 2 * u**2 + u) * durations[segment] * velocities[segment]
                + (-2 * u**3 + 3 * u**2) * points[segment + 1]
                + (u**3 - u**2) * durations[segment] * velocities[segment + 1]
            )
            target_ctrl[5] = ready_gripper
            advance_to_ctrl(model, data, window, target_ctrl, phase=label, observer=observer)


def smooth_gripper_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: object | None,
    target: float,
    ticks: int,
    cube_id: int | None = None,
    observer: Observer | None = None,
) -> None:
    """用 minimum-jerk 平滑开合夹爪；抓取时接触稳定后提前结束。

    阶段内只通过 advance_to_ctrl 下发命令，保证每个 tick 记录的动作就是真正执行的
    命令：不允许在 tick 之外直接写 data.ctrl，否则开环回放会在该处错拍。
    最小 jerk 序列可能被单 tick 限速截断（release 段尤其明显），因此循环之后继续用
    限速补齐到目标值，而不是一步跳过限速。
    """
    start = float(data.ctrl[5])
    contact_ticks = 0
    for fraction in np.linspace(0.0, 1.0, ticks)[1:]:
        progress = minimum_jerk(fraction)
        target_ctrl = data.ctrl.copy()
        target_ctrl[5] = start * (1.0 - progress) + target * progress
        advance_to_ctrl(model, data, window, target_ctrl, phase="gripper", observer=observer)
        if cube_id is not None and fraction >= 0.35:
            contacts = contact_report(model, data, cube_id)
            if {GRIPPER_BODY, "moving_jaw_so101_v1"}.issubset(contacts):
                contact_ticks += 1
                if contact_ticks == 1:
                    print("夹爪已检测到双侧接触，继续短暂收紧。")
                if contact_ticks >= 4:
                    break
    gated = 0
    while abs(float(data.ctrl[5]) - target) > 1e-12:
        target_ctrl = data.ctrl.copy()
        target_ctrl[5] = target
        advance_to_ctrl(model, data, window, target_ctrl, phase="gripper", observer=observer)
        gated += 1
        if gated > 1000:
            raise RuntimeError("夹爪目标在 1000 个 tick 内仍未收敛，检查限速配置")


def contact_report(model: mujoco.MjModel, data: mujoco.MjData, cube_id: int) -> list[str]:
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if cube_id not in (body1, body2):
            continue
        other = body2 if body1 == cube_id else body1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or str(other)
        contacts.append(name)
    return sorted(set(contacts))


def cube_in_target_box(model: mujoco.MjModel, data: mujoco.MjData, cube_id: int, target_id: int) -> bool:
    cube = data.xpos[cube_id]
    target = data.xpos[target_id]
    # 盒子内边界约为 ±50 mm，扣除 15 mm 方块半边长后，中心允许范围约为 ±35 mm。
    inside_xy = abs(cube[0] - target[0]) <= 0.035 and abs(cube[1] - target[1]) <= 0.035
    resting_on_table = 0.010 <= cube[2] <= 0.035
    return bool(inside_xy and resting_on_table)


@dataclass(frozen=True)
class Waypoints:
    """一次演示用到的关键点，抓取段和可达性预检共用同一套定义。"""

    pregrasp: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    above_target: np.ndarray
    place: np.ndarray


def task_waypoints(profile: MotionProfile, cube_start: np.ndarray, target_surface: np.ndarray) -> Waypoints:
    surface = np.asarray(target_surface, dtype=np.float64).copy()
    # 参考远端 planner 的 RELEASE_Z：从盒口上方释放，让方块自然落到底板。
    surface[2] = 0.080 if profile.name == "real_safe" else 0.055
    # real_safe 从盒内垂直抬高一些再撤回，给盒壁留出碰撞裕量。
    above_height = 0.045 if profile.name == "real_safe" else 0.065
    return Waypoints(
        pregrasp=cube_start + np.array([0.0, 0.0, 0.060]),
        grasp=cube_start + np.array([0.0, 0.0, 0.012]),
        lift=cube_start + np.array([0.0, 0.0, 0.100]),
        above_target=surface + np.array([0.0, 0.0, above_height]),
        place=surface + np.array([0.0, 0.0, 0.012]),
    )


# 可达性预检的分段误差上限：超过就丢弃样本，而不是执行到一半才失败。
#
# 抓取点、预抓取点和释放点决定动作几何是否正确，用 5 mm 硬标准；实测这三段误差都在
# 0.1 mm 以内，裕量充足。
# 「抬升」和「移动到盒子上方」只是让开障碍的中间路点，目标点位于 5 自由度手臂在
# 「末端朝下」约束下的灵巧工作空间之外：实测抬升段最优误差 6.9–20 mm（x=0.22 最小，
# x=0.25 为 13.5 mm），盒上方固定约 17.9 mm；用 300 次随机初值重启也找不到更优解，
# wrist_flex 已经顶到关节限位。move_to 本身按 30 mm 判定，这里取 25 mm，既接受这些
# 固有偏差，又能挡住在抬升阶段就发散（>30 mm）的样本。
REACHABILITY_TOLERANCE_M = {
    "接近": 0.005,
    "下降": 0.005,
    "抬升": 0.025,
    "移动到盒子上方": 0.025,
    "下降到盒内": 0.005,
}


@dataclass(frozen=True)
class ReachabilityReport:
    reachable: bool
    max_error_m: float
    failing_phase: str | None
    initial_collision: bool = False


def initial_arm_collision(model: mujoco.MjModel, data: mujoco.MjData, cube_id: int) -> list[str]:
    """复位后立即检查方块是否与手臂/夹爪穿模。

    folded_ready 姿态下夹爪停在桌面 x≈0.13–0.19 m、y≈0 的位置。把方块采样到这里，
    物理第一步就会把它弹开（实测穿透 4–20 mm，方块被推出 5 cm），整个 episode 必然
    失败，因此这种样本要在执行前丢弃。
    """
    return [name for name in contact_report(model, data, cube_id) if name != "world"]


def check_task_reachable(
    task: TaskSample | None,
    profile_name: str = "real_safe",
    tolerances_m: dict[str, float] | None = None,
) -> ReachabilityReport:
    """在真正执行之前，从复位状态出发依次预解 pregrasp → grasp → lift → above_target → place。

    每一段用上一段的结果作为初值，任何一段误差超过该段的上限就判定样本不可达；
    复位状态与手臂穿模的样本同样直接丢弃。
    """
    limits = REACHABILITY_TOLERANCE_M if tolerances_m is None else tolerances_m
    model, data = load_model(FIXED_SCENE_PATH)
    profile = get_motion_profile(profile_name)
    reset_to_ready(model, data, task)
    gripper_id = body_id(model, GRIPPER_BODY)
    cube_id = body_id(model, CUBE_BODY)
    target_id = body_id(model, TARGET_BOX_BODY)
    qpos_ids, dof_ids = arm_addresses(model)
    cube_start = data.xpos[cube_id].copy()
    if initial_arm_collision(model, data, cube_id):
        return ReachabilityReport(
            reachable=False,
            max_error_m=0.0,
            failing_phase="初始状态",
            initial_collision=True,
        )
    cube_rotation = np.asarray(data.xmat[cube_id]).reshape(3, 3)
    cube_yaw = float(np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0]))
    selected_yaw = select_square_grasp_yaw(cube_yaw, gripper_yaw(model, data, gripper_id))
    waypoints = task_waypoints(profile, cube_start, data.xpos[target_id])

    segments = (
        ("接近", waypoints.pregrasp, True),
        ("下降", waypoints.grasp, True),
        ("抬升", waypoints.lift, True),
        ("移动到盒子上方", waypoints.above_target, False),
        ("下降到盒内", waypoints.place, False),
    )
    worst_ratio = 0.0
    worst_error = 0.0
    failing_phase: str | None = None
    guess = data.qpos[qpos_ids].copy()
    for label, target, constrain_yaw in segments:
        data.qpos[qpos_ids] = guess
        q, error = solve_ik(
            model,
            data,
            target,
            qpos_ids,
            dof_ids,
            gripper_id,
            target_yaw=selected_yaw if constrain_yaw else None,
        )
        guess = q
        ratio = float(error) / limits[label]
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_error = float(error)
            failing_phase = label
    return ReachabilityReport(
        reachable=worst_ratio <= 1.0,
        max_error_m=worst_error,
        failing_phase=None if worst_ratio <= 1.0 else failing_phase,
    )


def run(
    with_viewer: bool,
    observer: Observer | None = None,
    profile_name: str = "simulation_fast",
    task: TaskSample | None = None,
) -> bool:
    model, data = load_model(FIXED_SCENE_PATH)
    profile = get_motion_profile(profile_name)
    configure_profile(profile, model.opt.timestep)
    reset_to_ready(model, data, task)

    gripper_id = body_id(model, GRIPPER_BODY)
    cube_id = body_id(model, CUBE_BODY)
    target_id = body_id(model, TARGET_BOX_BODY)
    qpos_ids, dof_ids = arm_addresses(model)
    ready_arm = data.qpos[:5].copy()
    ready_gripper = float(data.qpos[5])
    cube_start = data.xpos[cube_id].copy()
    cube_rotation = np.asarray(data.xmat[cube_id]).reshape(3, 3)
    cube_yaw = float(np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0]))
    yaw_candidates = square_grasp_yaw_candidates(cube_yaw)
    # 比较对象是夹爪闭合方向在世界坐标系中的 yaw（已经包含 shoulder_pan 的贡献），
    # 不是 wrist_roll 的关节角；关节角与世界 yaw 之间还差一个随姿态变化的偏移。
    current_yaw = gripper_yaw(model, data, gripper_id)
    selected_yaw = select_square_grasp_yaw(cube_yaw, current_yaw)
    waypoints = task_waypoints(profile, cube_start, data.xpos[target_id])
    window = None
    if with_viewer:
        from mujoco import viewer

        window = viewer.launch_passive(model, data)
        window.cam.lookat[:] = [0.15, 0.0, 0.10]
        window.cam.distance = 0.65
        window.cam.azimuth = 135
        window.cam.elevation = -25

    try:
        print(
            "初始姿态 → 打开夹爪 → 接近 → 下降 → 闭合 → 抬升 → 移到盒上方 → 放入 → 返回初始姿态",
            flush=True,
        )
        print(
            f"正方体等效 yaw 候选: {[round(yaw, 4) for yaw in yaw_candidates]}，"
            f"当前夹爪世界 yaw: {current_yaw:.4f} rad，当前选择: {selected_yaw:.4f} rad"
        )
        if window is not None:
            sync(window)
            time.sleep(0.5)
        step(model, data, window, profile.initial_physics_steps, phase="initial", observer=observer)
        pregrasp = waypoints.pregrasp
        grasp = waypoints.grasp
        # 抓取相关阶段锁定 yaw；放置和回程不约束 yaw，避免盒内姿态互相拉扯。
        move_to(
            model,
            data,
            window,
            pregrasp,
            qpos_ids,
            dof_ids,
            gripper_id,
            "接近",
            observer,
            target_yaw=selected_yaw,
        )
        move_to(
            model,
            data,
            window,
            grasp,
            qpos_ids,
            dof_ids,
            gripper_id,
            "下降",
            observer,
            target_yaw=selected_yaw,
        )

        smooth_gripper_to(
            model, data, window, GRIPPER_CLOSED, profile.close_ticks, cube_id=cube_id, observer=observer
        )
        step(
            model,
            data,
            window,
            profile.physics_per_tick * 10,
            phase="grasp_hold",
            observer=observer,
        )
        contacts = contact_report(model, data, cube_id)
        print(f"闭合后方块接触 body: {contacts or '无'}")

        lift = waypoints.lift
        move_to(
            model,
            data,
            window,
            lift,
            qpos_ids,
            dof_ids,
            gripper_id,
            "抬升",
            observer,
            target_yaw=selected_yaw,
        )
        step(model, data, window, profile.lift_hold_physics_steps, phase="lift_hold", observer=observer)
        lifted = float(data.xpos[cube_id, 2] - cube_start[2])
        final_contacts = contact_report(model, data, cube_id)
        bilateral_contact = {GRIPPER_BODY, "moving_jaw_so101_v1"}.issubset(final_contacts)
        grasp_ok = lifted >= 0.070 and bilateral_contact and "world" not in final_contacts
        print(f"方块抬升高度: {lifted:.4f} m")
        print(f"保持 1 秒后的方块接触 body: {final_contacts or '无'}")
        print(f"抓取判定: {'成功' if grasp_ok else '失败'}（阈值 0.0700 m）")

        above_target = waypoints.above_target
        place = waypoints.place
        move_to(model, data, window, above_target, qpos_ids, dof_ids, gripper_id, "移动到盒子上方", observer)
        move_to(model, data, window, place, qpos_ids, dof_ids, gripper_id, "下降到盒内", observer)

        smooth_gripper_to(model, data, window, GRIPPER_OPEN, profile.release_ticks, observer=observer)
        step(
            model,
            data,
            window,
            profile.release_settle_physics_steps,
            phase="release_settle",
            observer=observer,
        )
        placed = cube_in_target_box(model, data, cube_id, target_id)
        print(f"释放后方块位置: {np.array2string(data.xpos[cube_id], precision=4)}")
        print(f"盒内放置判定: {'成功' if placed else '失败'}")

        # 快速仿真保留末端撤回；real_safe 已在盒口上方释放，直接走关节空间 clearance。
        if profile.name == "simulation_fast":
            move_to(model, data, window, above_target, qpos_ids, dof_ids, gripper_id, "从盒口撤回", observer)
        return_to_ready(model, data, window, ready_arm, ready_gripper, profile, observer)
        data.ctrl[:5] = ready_arm
        data.ctrl[5] = ready_gripper
        arm_error = float(np.max(np.abs(data.qpos[:5] - ready_arm)))
        gripper_error = abs(float(data.qpos[5] - ready_gripper))
        returned = arm_error <= 0.05 and gripper_error <= 0.03
        print(
            f"返回初始姿态误差: arm={arm_error:.4f} rad, gripper={gripper_error:.4f} rad "
            f"(各关节 {np.array2string(data.qpos[:5] - ready_arm, precision=4)})"
        )
        print(f"初始姿态返回判定: {'成功' if returned else '失败'}（arm 阈值 0.0500 rad）")
        success = grasp_ok and placed and returned
        if not grasp_ok:
            failure_phase = "抬升"
            failure_reason = (
                f"抓取失败: 抬升 {lifted:.4f} m, 接触 {final_contacts or '无'}（阈值 0.0700 m 且双侧接触）"
            )
        elif not placed:
            failure_phase = "放置"
            failure_reason = f"盒内放置失败: 方块位置 {np.array2string(data.xpos[cube_id], precision=4)}"
        elif not returned:
            failure_phase = "回程"
            failure_reason = (
                f"回程失败: arm={arm_error:.4f} rad, gripper={gripper_error:.4f} rad"
                f"（阈值 0.0500/0.0300 rad）"
            )
        else:
            failure_phase = None
            failure_reason = None
        print(f"速度配置: {profile.name}")
        print(f"完整动作判定: {'成功' if success else '失败'}")
        if window is not None:
            print("演示已完成，窗口停留在最终姿态；关闭窗口即可退出。", flush=True)
            while window.is_running():
                window.sync()
                time.sleep(0.02)
        return RunResult(
            success=success,
            grasp_ok=grasp_ok,
            placed=placed,
            returned=returned,
            failure_phase=failure_phase,
            failure_reason=failure_reason,
        )
    except (RuntimeError, ValueError) as error:
        # IK 不收敛、yaw 不收敛等：不打断采集，转成结构化失败结果。
        print(f"演示中断于阶段 {CURRENT_PHASE}: {error}")
        return RunResult(
            success=False,
            grasp_ok=False,
            placed=False,
            returned=False,
            failure_phase=CURRENT_PHASE,
            failure_reason=str(error),
        )
    finally:
        if window is not None:
            window.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true", help="只运行物理仿真，不打开窗口")
    parser.add_argument(
        "--profile",
        choices=("simulation_fast", "real_safe"),
        default="simulation_fast",
        help="动作速度配置；real_safe 用于真机准备阶段的保守仿真",
    )
    parser.add_argument("--seed", type=int, default=None, help="按采样器随机一个任务初始状态")
    parser.add_argument(
        "--cube-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="直接指定方块水平位置（米）",
    )
    parser.add_argument("--cube-yaw-deg", type=float, default=None, help="直接指定方块 yaw（度）")
    args = parser.parse_args()

    task = None
    if args.seed is not None:
        task = sample_task(derive_seed(args.seed, 0), SamplerConfig())
    elif args.cube_xy is not None or args.cube_yaw_deg is not None:
        cube_xy = tuple(args.cube_xy) if args.cube_xy is not None else (0.25, 0.0)
        cube_yaw = np.deg2rad(args.cube_yaw_deg) if args.cube_yaw_deg is not None else 0.0
        task = TaskSample(cube_xy=cube_xy, cube_yaw=float(cube_yaw), box_xy=SCENE_BOX_XY, seed=0)
    if task is not None:
        print(
            f"任务采样: cube_xy=({task.cube_xy[0]:.4f}, {task.cube_xy[1]:.4f}) "
            f"cube_yaw={task.cube_yaw_deg:.2f}° box_xy={task.box_xy} seed={task.seed}"
        )
    try:
        result = run(not args.no_viewer, profile_name=args.profile, task=task)
        if not result.success:
            print(f"失败阶段: {result.failure_phase}，原因: {result.failure_reason}")
        return 0 if result.success else 1
    except InterruptedError as error:
        print(str(error))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
