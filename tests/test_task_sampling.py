"""验证任务采样：范围、可复现性、分层覆盖和 apply_task 的写入结果。"""

from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np
import pytest

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model
from task_sampling import (
    SCENE_BOX_XY,
    SamplerConfig,
    TaskSample,
    apply_task,
    cell_bounds,
    cell_count,
    cell_for_index,
    derive_seed,
    sample_task,
)


def test_scene_box_constant_matches_model():
    model, _ = load_model(FIXED_SCENE_PATH)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    assert np.allclose(model.body_pos[box_id][:2], SCENE_BOX_XY, atol=1e-12)


def test_sample_is_reproducible_and_within_range():
    cfg = SamplerConfig()
    first = sample_task(12345, cfg)
    second = sample_task(12345, cfg)
    assert first == second
    assert first.seed == 12345
    for seed in range(64):
        sample = sample_task(seed, cfg)
        assert cfg.radius_m[0] <= sample.radius_m <= cfg.radius_m[1]
        assert cfg.azimuth_deg[0] <= sample.azimuth_deg <= cfg.azimuth_deg[1]
        assert cfg.cube_yaw_deg[0] <= sample.cube_yaw_deg <= cfg.cube_yaw_deg[1]
        assert sample.box_xy == SCENE_BOX_XY  # 默认不动盒子


def test_samples_cover_the_configured_ranges():
    samples = [sample_task(seed) for seed in range(200)]
    radii = [s.radius_m for s in samples]
    azimuths = [s.azimuth_deg for s in samples]
    yaws = [s.cube_yaw_deg for s in samples]
    assert min(radii) < 0.205 and max(radii) > 0.235
    assert min(azimuths) < -25.0 and max(azimuths) > 25.0
    assert min(yaws) < 5.0 and max(yaws) > 85.0


def test_default_radius_excludes_the_calibrated_bad_region():
    """标定结论：r < 0.20 m 时专家会抓不住方块，默认范围必须排除它。"""
    assert SamplerConfig.radius_m[0] >= 0.20
    assert SamplerConfig.radius_m[1] <= 0.24


def test_box_jitter_moves_box_within_range():
    cfg = SamplerConfig(box_jitter_m=0.01)
    sample = sample_task(7, cfg)
    assert abs(sample.box_xy[0] - SCENE_BOX_XY[0]) <= 0.01
    assert abs(sample.box_xy[1] - SCENE_BOX_XY[1]) <= 0.01
    assert sample.box_xy != SCENE_BOX_XY


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        sample_task(0, SamplerConfig(radius_m=(0.3, 0.2)))
    with pytest.raises(ValueError):
        sample_task(0, SamplerConfig(azimuth_deg=(30.0, -30.0)))
    with pytest.raises(ValueError):
        sample_task(0, SamplerConfig(cube_yaw_deg=(90.0, 0.0)))
    with pytest.raises(ValueError):
        sample_task(0, SamplerConfig(box_jitter_m=-0.1))


def test_apply_task_writes_cube_pose_without_touching_scene():
    model, data = load_model(FIXED_SCENE_PATH)
    sample = TaskSample(cube_xy=(0.20, -0.11), cube_yaw=np.deg2rad(37.0), box_xy=SCENE_BOX_XY, seed=1)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_id = int(model.jnt_qposadr[cube_joint_id])
    height_before = float(data.qpos[cube_qpos_id + 2])

    apply_task(model, data, sample)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert np.allclose(data.xpos[cube_id][:2], sample.cube_xy, atol=1e-9)
    # 高度沿用 keyframe，不要在空中生成方块。
    assert data.xpos[cube_id][2] == pytest.approx(height_before, abs=1e-9)
    assert data.xpos[cube_id][2] == pytest.approx(0.015, abs=1e-9)
    rotation = np.asarray(data.xmat[cube_id]).reshape(3, 3)
    assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(sample.cube_yaw, abs=1e-9)
    # 盒子位置不变时模型不应被改动。
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    assert np.allclose(data.xpos[box_id][:2], SCENE_BOX_XY, atol=1e-9)


def test_apply_task_moves_box_when_jittered():
    model, data = load_model(FIXED_SCENE_PATH)
    sample = TaskSample(cube_xy=(0.20, 0.0), cube_yaw=0.0, box_xy=(0.05, 0.19), seed=2)
    apply_task(model, data, sample)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    assert np.allclose(data.xpos[box_id][:2], sample.box_xy, atol=1e-9)


def test_derived_seed_is_stable_and_unique_per_index():
    seeds = [derive_seed(20261010, index) for index in range(256)]
    assert len(set(seeds)) == len(seeds), "不同编号不能拿到相同种子"
    # 续采时同一个编号必须还是同一个种子，与本次采集从哪里开始无关。
    assert derive_seed(20261010, 137) == derive_seed(20261010, 137)
    # 重采样用不同的子种子。
    assert derive_seed(20261010, 137, 0) != derive_seed(20261010, 137, 1)
    assert derive_seed(20261010, 137) != derive_seed(20261011, 137)


# --------------------------------------------------------------------------------------
# 分层采样
# --------------------------------------------------------------------------------------


def stratified() -> SamplerConfig:
    return SamplerConfig(mode="stratified")


def test_stratified_has_64_cells_by_default():
    assert cell_count(stratified()) == 64


def test_stratified_first_block_covers_every_cell_exactly_once():
    """前 64 条（= 格子总数）每格恰好 1 条。"""
    cfg = stratified()
    cells = [cell_for_index(index, cfg) for index in range(cell_count(cfg))]
    assert sorted(cells) == list(range(64))


@pytest.mark.parametrize("count", [1, 7, 50, 63, 64, 65, 100, 200, 300])
def test_stratified_prefix_is_balanced(count):
    """任意前 N 条里，每格的条数最多差 1（没轮到的格子按 0 计）。"""
    cfg = stratified()
    tally = Counter(cell_for_index(index, cfg) for index in range(count))
    counts = [tally.get(cell, 0) for cell in range(cell_count(cfg))]
    assert max(counts) - min(counts) <= 1


def test_stratified_same_index_reproduces_the_same_task():
    cfg = stratified()
    index = 37
    cell = cell_for_index(index, cfg)
    first = sample_task(derive_seed(20261010, index, 0), cfg, cell=cell)
    second = sample_task(derive_seed(20261010, index, 0), cfg, cell=cell)
    assert first == second
    assert first.cell == cell


def test_stratified_resampling_stays_inside_the_same_cell():
    """丢弃重采只换派生种子，不能跳到下一格。"""
    cfg = stratified()
    index = 51
    cell = cell_for_index(index, cfg)
    (r_low, r_high), (az_low, az_high), (yaw_low, yaw_high) = cell_bounds(cell, cfg)
    samples = [sample_task(derive_seed(20261010, index, attempt), cfg, cell=cell) for attempt in range(8)]
    for sample in samples:
        assert sample.cell == cell
        assert r_low <= sample.radius_m <= r_high
        assert az_low <= sample.azimuth_deg <= az_high
        assert yaw_low <= sample.cube_yaw_deg <= yaw_high
    # 每次 attempt 都换了种子，样本不应全部相同。
    assert len({(s.cube_xy, s.cube_yaw) for s in samples}) == len(samples)


def test_stratified_cells_tile_the_calibrated_ranges():
    """64 格的并集正好铺满标定区间，格与格之间首尾相接。"""
    cfg = stratified()
    bounds = [cell_bounds(cell, cfg) for cell in range(cell_count(cfg))]
    for axis, configured in enumerate((cfg.radius_m, cfg.azimuth_deg, cfg.cube_yaw_deg)):
        lows = sorted(bound[axis][0] for bound in bounds)
        highs = sorted(bound[axis][1] for bound in bounds)
        assert lows[0] == pytest.approx(configured[0])
        assert highs[-1] == pytest.approx(configured[1])
        # 4 档 → 每档边界在区间内出现 4 次
        assert len(set(np.round(lows, 12))) == 4
        assert len(set(np.round(highs, 12))) == 4


def test_stratified_needs_a_cell_and_uniform_refuses_one():
    with pytest.raises(ValueError, match="必须给出 cell"):
        sample_task(1, stratified())
    with pytest.raises(ValueError, match="不接受 cell"):
        sample_task(1, SamplerConfig(), cell=0)


def test_unknown_mode_and_bad_bins_are_rejected():
    with pytest.raises(ValueError, match="未知采样模式"):
        sample_task(1, SamplerConfig(mode="zigzag"))
    with pytest.raises(ValueError, match="分层档数"):
        cell_count(SamplerConfig(mode="stratified", bins=(4, 0, 4)))
    with pytest.raises(ValueError, match="不在"):
        cell_bounds(64, stratified())


def test_negative_index_is_rejected():
    with pytest.raises(ValueError, match="编号不能为负"):
        cell_for_index(-1, stratified())
