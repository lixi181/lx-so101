"""抓取任务初始状态采样。

采样用极坐标：半径和方位角分别在一个区间内均匀采样，方块 yaw 利用正方体的 90° 对称性
只在 [0°, 90°) 内采样。

半径范围由 5×5×4 网格标定得到（`scripts/calibrate_workspace.py`，报告见
`results/workspace_calibration_full.json`）：最初按探测结果取 r ∈ (0.16, 0.24)，
实测 100 个格子里 74 个成功、22 个在执行前被丢弃，可达样本内成功率 94.9%，不满足
95% 的要求；其中 4 次真正的专家失败全部出现在 r ≤ 0.18 的格子上（抬升阶段没夹住
方块）。缩小到 r ∈ (0.20, 0.24) 后，60 个格子里执行了 54 次全部成功（100%），
6 次因初始状态穿模或 IK 超限被预检丢弃。因此默认范围取 (0.20, 0.24)。

采样结果 `TaskSample` 是纯数据，写入 episode.json 后可以完整复现同一个初始状态：
`apply_task` 只写入 cube freejoint 的 qpos 和（可选的）盒子位置，场景 XML 与
keyframe 保持不变。

两种采样模式：

- `uniform`：在区间内完全均匀随机，`sample_task(seed)` 直接可用；
- `stratified`（采集默认）：r/θ/yaw 各切成 `bins` 档（默认 4×4×4 = 64 格），格子按
  `编号 % 64` 轮转分配，所以任意前 N 条里每格条数最多差 1。格子内均匀抖动。
  丢弃重采必须传同一个 `cell` 配新的派生种子，这样重采不会跳到别的格子。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# scene_fixed.xml 中 target_box 的默认位置，box_jitter_m 为 0 时盒子不动。
# tests/test_task_sampling.py 会校验这个常量与模型一致，避免场景改了而常量没改。
SCENE_BOX_XY: tuple[float, float] = (0.04, 0.18)
CUBE_FREEJOINT_NAME = "cube_freejoint"
TARGET_BOX_BODY = "target_box"


@dataclass(frozen=True)
class SamplerConfig:
    """任务采样范围。整份配置会写进 episode.json。"""

    # 标定结论：r < 0.20 m 时专家会抓不住方块。
    radius_m: tuple[float, float] = (0.20, 0.24)
    azimuth_deg: tuple[float, float] = (-30.0, 30.0)
    cube_yaw_deg: tuple[float, float] = (0.0, 90.0)  # 正方体有 90° 对称，这个范围就够了
    box_jitter_m: float = 0.0  # 第一版不动盒子
    # 采样模式与分层档数（r、θ、yaw 各几档，相乘就是格子总数）。
    # 采集默认用 stratified（见 `collect_pick_place_dataset.py` 的 `--sampler-mode`）；
    # 这里默认 uniform，是为了让 `sample_task(seed)` 这种不指定格子的调用继续可用。
    mode: str = "uniform"
    bins: tuple[int, int, int] = (4, 4, 4)


SAMPLER_MODES = ("uniform", "stratified")


@dataclass(frozen=True)
class TaskSample:
    """一次性采样的完整任务初始状态。"""

    cube_xy: tuple[float, float]
    cube_yaw: float
    box_xy: tuple[float, float]
    seed: int
    # 分层采样的格子编号；uniform 模式为 None。写进 episode.json 便于反查覆盖情况。
    cell: int | None = None

    @property
    def cube_yaw_deg(self) -> float:
        return float(np.rad2deg(self.cube_yaw))

    @property
    def radius_m(self) -> float:
        return float(np.hypot(*self.cube_xy))

    @property
    def azimuth_deg(self) -> float:
        return float(np.rad2deg(np.arctan2(self.cube_xy[1], self.cube_xy[0])))


def derive_seed(base_seed: int, episode_index: int, attempt: int = 0) -> int:
    """由 (基础种子, 编号, 第几次重采样) 推导种子。

    同一个编号永远对应同一个种子，与分几次采集、从哪里续采无关；`attempt` 用于
    采样被可达性检查拒绝后重新采样。
    """
    entropy = [int(base_seed), int(episode_index), int(attempt)]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


DEFAULT_SAMPLER_CONFIG = SamplerConfig()


def bins_of(cfg: SamplerConfig) -> tuple[int, int, int]:
    """校验并返回分层档数。"""
    bins = tuple(int(value) for value in cfg.bins)
    if len(bins) != 3 or any(value < 1 for value in bins):
        raise ValueError(f"分层档数非法: {cfg.bins}")
    return bins  # type: ignore[return-value]


def cell_count(cfg: SamplerConfig) -> int:
    bins = bins_of(cfg)
    return bins[0] * bins[1] * bins[2]


def cell_for_index(index: int, cfg: SamplerConfig) -> int:
    """按编号轮转分配格子。

    取模即可满足「任意前 N 条在各格之间大致均衡、每格数量最多差 1」：前 64 条
    （格子总数）每格恰好 1 条。丢弃样本重采时必须传同一个 `cell`，只换派生种子，
    所以重采落在同一格里。
    """
    if index < 0:
        raise ValueError(f"编号不能为负: {index}")
    return index % cell_count(cfg)


def cell_bins(cell: int, cfg: SamplerConfig) -> tuple[int, int, int]:
    """把格子编号拆成 (r 档, θ 档, yaw 档)。yaw 是最内层，保证相邻编号只挪 yaw 档。"""
    bins = bins_of(cfg)
    total = cell_count(cfg)
    if not 0 <= cell < total:
        raise ValueError(f"格子编号 {cell} 不在 [0, {total}) 内")
    bin_yaw = cell % bins[2]
    bin_azimuth = (cell // bins[2]) % bins[1]
    bin_radius = cell // (bins[1] * bins[2])
    return bin_radius, bin_azimuth, bin_yaw


def cell_bounds(
    cell: int, cfg: SamplerConfig
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """格子在 (r, θ, yaw) 三个方向上的区间。"""
    dims = (cfg.radius_m, cfg.azimuth_deg, cfg.cube_yaw_deg)
    bounds = []
    for bin_index, count, (low, high) in zip(cell_bins(cell, cfg), bins_of(cfg), dims, strict=True):
        width = (high - low) / count
        bounds.append((low + width * bin_index, low + width * (bin_index + 1)))
    return bounds[0], bounds[1], bounds[2]


def sample_task(
    seed: int, cfg: SamplerConfig = DEFAULT_SAMPLER_CONFIG, cell: int | None = None
) -> TaskSample:
    """按极坐标采样一个任务初始状态。

    `mode="stratified"` 时必须给出 `cell`：先把 r/θ/yaw 的区间各切成 `bins` 档，
    格子内均匀抖动。重采样传同一个 `cell` 加新的 `seed`，就落在同一格里。
    """
    radius_low, radius_high = cfg.radius_m
    azimuth_low, azimuth_high = cfg.azimuth_deg
    yaw_low, yaw_high = cfg.cube_yaw_deg
    if not 0.0 < radius_low <= radius_high:
        raise ValueError(f"半径范围非法: {cfg.radius_m}")
    if azimuth_low > azimuth_high:
        raise ValueError(f"方位角范围非法: {cfg.azimuth_deg}")
    if yaw_low > yaw_high:
        raise ValueError(f"方块 yaw 范围非法: {cfg.cube_yaw_deg}")
    if cfg.box_jitter_m < 0.0:
        raise ValueError(f"盒子抖动必须非负: {cfg.box_jitter_m}")
    if cfg.mode not in SAMPLER_MODES:
        raise ValueError(f"未知采样模式: {cfg.mode!r}；可选: {', '.join(SAMPLER_MODES)}")

    if cfg.mode == "stratified":
        if cell is None:
            raise ValueError('mode="stratified" 时必须给出 cell')
        (r_low, r_high), (az_low, az_high), (yaw_lo, yaw_hi) = cell_bounds(cell, cfg)
    else:
        if cell is not None:
            raise ValueError('mode="uniform" 不接受 cell')
        (r_low, r_high) = (radius_low, radius_high)
        (az_low, az_high) = (azimuth_low, azimuth_high)
        (yaw_lo, yaw_hi) = (yaw_low, yaw_high)

    # 三个量的抽取顺序固定（半径 → 方位角 → yaw → 盒子抖动），换模式不会打乱随机流。
    rng = np.random.default_rng(seed)
    radius = float(rng.uniform(r_low, r_high))
    azimuth = float(np.deg2rad(rng.uniform(az_low, az_high)))
    cube_yaw = float(np.deg2rad(rng.uniform(yaw_lo, yaw_hi)))
    box_xy = (
        float(SCENE_BOX_XY[0] + rng.uniform(-cfg.box_jitter_m, cfg.box_jitter_m)),
        float(SCENE_BOX_XY[1] + rng.uniform(-cfg.box_jitter_m, cfg.box_jitter_m)),
    )
    return TaskSample(
        cube_xy=(radius * float(np.cos(azimuth)), radius * float(np.sin(azimuth))),
        cube_yaw=cube_yaw,
        box_xy=box_xy,
        seed=int(seed),
        cell=cell,
    )


def apply_task(model: mujoco.MjModel, data: mujoco.MjData, sample: TaskSample) -> None:
    """在 mj_resetDataKeyframe 之后写入 cube freejoint 的 qpos，然后调用 mj_forward。

    只改方块的水平位置、方块 yaw 和盒子位置；方块高度沿用 keyframe（约 0.015 m），
    这样每次任务都是从桌面上开始，而不是从空中落下。
    """
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_FREEJOINT_NAME)
    if cube_joint_id < 0:
        raise ValueError(f"找不到 freejoint: {CUBE_FREEJOINT_NAME}")
    cube_qpos_id = int(model.jnt_qposadr[cube_joint_id])
    data.qpos[cube_qpos_id : cube_qpos_id + 2] = sample.cube_xy
    half = sample.cube_yaw / 2.0
    data.qpos[cube_qpos_id + 3 : cube_qpos_id + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BOX_BODY)
    if box_id < 0:
        raise ValueError(f"找不到 body: {TARGET_BOX_BODY}")
    model.body_pos[box_id, :2] = sample.box_xy

    mujoco.mj_forward(model, data)
