"""生成第一版评估布局：只产出布局，不采集、不渲染。

50 个布局沿用训练那套分层采样，但格子是**在整个 64 格里均匀摊开**的（而不是像训练那样
从 0 开始按编号轮转），所以评估覆盖的是整个工作空间，不会挤在同一角。

种子空间与训练分开：评估用 `EVAL_BASE_SEED`，训练用采集器默认的 `TRAIN_BASE_SEED`。
脚本会断言评估用的每一个派生种子都不出现在训练前 `TRAIN_OVERLAP_CHECK_INDICES` 个编号
（× 每次尝试）的种子集合里。

输出是**确定性**的：内容只由种子、配置和可达性预检决定，不含时间戳或 commit 号，所以
重复运行得到逐字节相同的文件、同一个 sha256 —— README 里记的哈希因此可以被复验。

布局文件按约定不进 Git（`datasets/evaluation/*` 已在 .gitignore 里）。

用法：

    ./.venv/bin/python scripts/generate_eval_layouts.py
    ./.venv/bin/python scripts/generate_eval_layouts.py --count 64
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from pick_cube import check_task_reachable

from task_sampling import SamplerConfig, cell_count, derive_seed, sample_task

# 训练种子基准必须与采集器的 --seed 默认值一致；tests/test_eval_layouts.py 会校验。
TRAIN_BASE_SEED = 20261010
# 评估种子基准：与训练不同的 base seed，派生出的种子集合与训练无交集。
EVAL_BASE_SEED = 20261011
TRAIN_OVERLAP_CHECK_INDICES = 10000
# 单个布局最多重采几次；每次尝试换派生种子但**不换格子**。
MAX_ATTEMPTS = 16
DEFAULT_COUNT = 50
DEFAULT_PROFILE = "real_safe"
FORMAT_VERSION = "local_raw_v3"
DEFAULT_OUTPUT = Path("datasets/evaluation/eval_layouts_v1.json")


def spread_cells(count: int, total: int) -> list[int]:
    """把 count 个布局尽量均匀地摊到 total 个格子上，返回升序的格子编号。

    `(index * total) // count` 在 count <= total 时严格递增，因此不会有重复格子，
    未覆盖的格子也分散在整个空间里，而不是集中在一角。
    """
    if count < 1:
        raise ValueError(f"布局数必须大于 0: {count}")
    if count > total:
        raise ValueError(f"布局数 {count} 超过格子总数 {total}")
    cells = [(index * total) // count for index in range(count)]
    if len(set(cells)) != count:
        raise ValueError(f"摊开格子时出现重复: {cells}")
    return cells


def train_seed_set(indices: int, attempts: int = MAX_ATTEMPTS) -> set[int]:
    """训练侧会用到的全部种子（base seed × 编号 × 尝试）。"""
    return {
        derive_seed(TRAIN_BASE_SEED, index, attempt)
        for index in range(indices)
        for attempt in range(attempts)
    }


def assert_seed_space_is_disjoint(layout_count: int, attempts: int = MAX_ATTEMPTS) -> int:
    """断言评估种子与训练种子不重叠，返回检查过的训练种子数。"""
    train = train_seed_set(TRAIN_OVERLAP_CHECK_INDICES, attempts)
    evaluation = {
        derive_seed(EVAL_BASE_SEED, index, attempt)
        for index in range(layout_count)
        for attempt in range(attempts)
    }
    overlap = evaluation & train
    if overlap:
        raise SystemExit(
            f"评估种子与训练前 {TRAIN_OVERLAP_CHECK_INDICES} 个编号的种子重叠 "
            f"{len(overlap)} 个，需要换一个 EVAL_BASE_SEED：{sorted(overlap)[:5]}"
        )
    return len(train)


def build_layouts(cfg: SamplerConfig, profile: str, count: int) -> tuple[list[dict], list[int]]:
    """逐格采样并做可达性预检；不可达就在同一格里换派生种子重采。"""
    cells = spread_cells(count, cell_count(cfg))
    layouts: list[dict] = []
    for layout_index, cell in enumerate(cells):
        for attempt in range(MAX_ATTEMPTS):
            seed = derive_seed(EVAL_BASE_SEED, layout_index, attempt)
            candidate = sample_task(seed, cfg, cell=cell)
            report = check_task_reachable(candidate, profile)
            if report.reachable:
                layouts.append(
                    {
                        "layout_index": layout_index,
                        "cell": cell,
                        "attempts": attempt,
                        "reachability_error_m": report.max_error_m,
                        "sample": dataclasses.asdict(candidate),
                    }
                )
                break
        else:
            raise SystemExit(
                f"布局 {layout_index}（cell {cell}）在 {MAX_ATTEMPTS} 次重采后仍不可达，未生成任何文件"
            )
    return layouts, cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--radius-min", type=float, default=SamplerConfig.radius_m[0])
    parser.add_argument("--radius-max", type=float, default=SamplerConfig.radius_m[1])
    parser.add_argument("--azimuth-min", type=float, default=SamplerConfig.azimuth_deg[0])
    parser.add_argument("--azimuth-max", type=float, default=SamplerConfig.azimuth_deg[1])
    parser.add_argument("--yaw-min", type=float, default=SamplerConfig.cube_yaw_deg[0])
    parser.add_argument("--yaw-max", type=float, default=SamplerConfig.cube_yaw_deg[1])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = SamplerConfig(
        radius_m=(args.radius_min, args.radius_max),
        azimuth_deg=(args.azimuth_min, args.azimuth_max),
        cube_yaw_deg=(args.yaw_min, args.yaw_max),
        mode="stratified",
    )
    checked = assert_seed_space_is_disjoint(args.count)
    print(
        f"种子空间检查通过：评估 base seed {EVAL_BASE_SEED} 与训练 base seed "
        f"{TRAIN_BASE_SEED} 前 {TRAIN_OVERLAP_CHECK_INDICES} 个编号的 {checked} 个种子无交集"
    )

    layouts, cells = build_layouts(cfg, args.profile, args.count)
    report = {
        "name": "eval_layouts_v1",
        "generated_by": "scripts/generate_eval_layouts.py",
        "format": FORMAT_VERSION,
        "profile": args.profile,
        "train_base_seed": TRAIN_BASE_SEED,
        "eval_base_seed": EVAL_BASE_SEED,
        "train_overlap_check_indices": TRAIN_OVERLAP_CHECK_INDICES,
        "sampler": dataclasses.asdict(cfg),
        "count": len(layouts),
        "cells_total": cell_count(cfg),
        "cells_covered": len(cells),
        "cells": cells,
        "layouts": layouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    retried = sum(1 for layout in layouts if layout["attempts"])
    print(f"生成 {len(layouts)} 个布局，覆盖 {len(cells)}/{cell_count(cfg)} 个格子")
    print(f"  其中重采过的布局: {retried} 个（不可达样本只在同一格内重采）")
    print(f"  可达性预检最差误差: {max(l['reachability_error_m'] for l in layouts) * 1000:.2f} mm")
    print(f"  未覆盖的格子: {sorted(set(range(cell_count(cfg))) - set(cells))}")
    print(f"写入 {args.output}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
