"""SO-101 仿真与真机准备阶段使用的动作速度配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionProfile:
    """动作时序和控制目标变化上限。

    `real_safe` 是真机准备阶段的保守仿真参数，不代表已经完成真机标定。
    """

    name: str
    physics_per_tick: int
    cartesian_ticks: int
    close_ticks: int
    release_ticks: int
    clearance_ticks: int
    folded_ready_ticks: int
    initial_physics_steps: int
    lift_hold_physics_steps: int
    release_settle_physics_steps: int
    max_joint_velocity_rad_s: float
    max_gripper_velocity_rad_s: float


SIMULATION_FAST = MotionProfile(
    name="simulation_fast",
    physics_per_tick=8,
    cartesian_ticks=36,
    close_ticks=36,
    release_ticks=18,
    clearance_ticks=50,
    folded_ready_ticks=70,
    # 以物理步计的时长按原来的秒数换算，再四舍五入到 physics_per_tick 的整数倍
    # （8 步/tick，timestep=1/480 s）。必须是整数倍：保持阶段按 tick 记录观测，
    # 不接受半截 tick。原值是按 0.002 s/步 定的：
    #   initial 120 步 = 0.240 s → 115.20 步 → 112 步（14 tick，0.23333 s，-2.8%）
    #   lift_hold 256 步 = 0.512 s → 245.76 步 → 248 步（31 tick，0.51667 s，+0.9%）
    #   release_settle 200 步 = 0.400 s → 192.00 步 → 192 步（24 tick，0.40000 s，恰好相等）
    initial_physics_steps=112,
    lift_hold_physics_steps=248,
    release_settle_physics_steps=192,
    max_joint_velocity_rad_s=8.0,
    max_gripper_velocity_rad_s=4.0,
)


REAL_SAFE = MotionProfile(
    name="real_safe",
    physics_per_tick=16,
    cartesian_ticks=64,
    close_ticks=30,
    release_ticks=15,
    clearance_ticks=60,
    folded_ready_ticks=90,
    # 同上，按原来的秒数换算再四舍五入到 16 的整数倍（16 步/tick，timestep=1/480 s，
    # 一个 tick 恰好 1/30 s）：
    #   initial 160 步 = 0.320 s → 153.60 步 → 160 步（10 tick，0.33333 s，+4.2%）
    #   lift_hold 320 步 = 0.640 s → 307.20 步 → 304 步（19 tick，0.63333 s，-1.0%）
    #   release_settle 240 步 = 0.480 s → 230.40 步 → 224 步（14 tick，0.46667 s，-2.8%）
    # 保持阶段按 tick 记录观测，所以这些值必须是 physics_per_tick 的整数倍。
    initial_physics_steps=160,
    lift_hold_physics_steps=304,
    release_settle_physics_steps=224,
    max_joint_velocity_rad_s=1.5,
    max_gripper_velocity_rad_s=1.5,
)


MOTION_PROFILES = {
    SIMULATION_FAST.name: SIMULATION_FAST,
    REAL_SAFE.name: REAL_SAFE,
}


def get_motion_profile(name: str) -> MotionProfile:
    try:
        return MOTION_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(sorted(MOTION_PROFILES))
        raise ValueError(f"未知 motion profile: {name}；可选: {choices}") from error
