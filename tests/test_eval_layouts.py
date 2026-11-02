"""验证评估布局生成：格子摊开、种子空间与训练分离、可达性与确定性。

布局文件本身不进 Git，所以这里测的是生成逻辑，不依赖那份产物。
"""

from __future__ import annotations

import itertools

import pytest
from module_loader import load_script

GENERATOR_SCRIPT = "scripts/generate_eval_layouts.py"
COLLECTOR_SCRIPT = "scripts/collect_pick_place_dataset.py"


@pytest.fixture
def generator():
    return load_script(GENERATOR_SCRIPT, "generate_eval_layouts_test")


@pytest.fixture
def collector():
    return load_script(COLLECTOR_SCRIPT, "collect_pick_place_dataset_for_eval_test")


def test_train_base_seed_matches_the_collector_default(generator, collector):
    """两边基准种子必须一致，否则「训练/评估种子分离」的断言就没有意义。"""
    default = collector.build_parser().get_default("seed")
    assert generator.TRAIN_BASE_SEED == default


def test_eval_base_seed_is_a_different_seed(generator):
    assert generator.EVAL_BASE_SEED != generator.TRAIN_BASE_SEED


def test_spread_cells_returns_distinct_cells(generator):
    cells = generator.spread_cells(50, 64)
    assert len(cells) == 50
    assert len(set(cells)) == 50
    assert all(0 <= cell < 64 for cell in cells)


def test_spread_cells_reaches_both_ends_and_stays_spread_out(generator):
    """摊开而不是挤在一角：两端都要覆盖到，且不留下大的连续空洞。"""
    cells = generator.spread_cells(50, 64)
    assert cells[0] == 0
    assert cells[-1] >= 63 - 2
    gaps = [b - a for a, b in itertools.pairwise(cells)]
    assert max(gaps) <= 3, f"出现 {max(gaps)} 格连续空洞，未覆盖区过于集中"


def test_full_count_covers_every_cell(generator):
    assert generator.spread_cells(64, 64) == list(range(64))


def test_spread_cells_rejects_impossible_counts(generator):
    with pytest.raises(ValueError, match="必须大于 0"):
        generator.spread_cells(0, 64)
    with pytest.raises(ValueError, match="超过格子总数"):
        generator.spread_cells(65, 64)


def test_eval_and_train_seed_spaces_are_disjoint(generator):
    """断言本身要能跑通，并且覆盖到前 10000 个训练编号。"""
    assert generator.TRAIN_OVERLAP_CHECK_INDICES == 10000
    checked = generator.assert_seed_space_is_disjoint(generator.DEFAULT_COUNT)
    # 返回值是**去重后**的训练种子数。160000 次派生里按生日问题预期约有
    # 160000²/2/2³² ≈ 3 次碰撞，所以略小于 10000 × 16 是正常的。
    expected = generator.TRAIN_OVERLAP_CHECK_INDICES * generator.MAX_ATTEMPTS
    assert expected - 10 <= checked <= expected


def test_disjoint_check_detects_a_collision(generator, monkeypatch):
    """把评估基准换成训练基准时必须报冲突，否则这个断言是空转的。"""
    monkeypatch.setattr(generator, "EVAL_BASE_SEED", generator.TRAIN_BASE_SEED)
    with pytest.raises(SystemExit, match="重叠"):
        generator.assert_seed_space_is_disjoint(4)


def test_layouts_are_reachable_and_use_distinct_cells(generator):
    cfg = generator.SamplerConfig(mode="stratified")
    layouts, cells = generator.build_layouts(cfg, "real_safe", 8)
    assert len(layouts) == 8
    assert len(set(cells)) == 8
    assert [layout["cell"] for layout in layouts] == cells
    for layout in layouts:
        assert layout["sample"]["cell"] == layout["cell"]
        assert layout["reachability_error_m"] >= 0.0
        assert layout["attempts"] < generator.MAX_ATTEMPTS


def test_layouts_are_deterministic(generator):
    """同一输入两次生成必须完全一致，否则 README 里记的 sha256 无法复验。"""
    cfg = generator.SamplerConfig(mode="stratified")
    first, _ = generator.build_layouts(cfg, "real_safe", 5)
    second, _ = generator.build_layouts(cfg, "real_safe", 5)
    assert first == second
