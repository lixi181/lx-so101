"""在工作空间网格上探测专家成功率，用于标定任务采样范围。

在 r × θ × yaw 上跑 5×5×4 的网格：每个格子先做可达性预检，不可达的直接记为丢弃，
可达的执行一次完整专家演示。最终成功率用来决定 `SamplerConfig` 的范围，结论写进 README。

用法：

    ./.venv/bin/python scripts/calibrate_workspace.py
    ./.venv/bin/python scripts/calibrate_workspace.py --output results/calibration.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
from pathlib import Path

import numpy as np
from pick_cube import RunResult, check_task_reachable, run

from task_sampling import SCENE_BOX_XY, TaskSample

DEFAULT_RADII_M = (0.16, 0.18, 0.20, 0.22, 0.24)
DEFAULT_AZIMUTHS_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
DEFAULT_YAWS_DEG = (0.0, 22.5, 45.0, 67.5)


def probe_cell(radius_m: float, azimuth_deg: float, yaw_deg: float, profile_name: str, verbose: bool) -> dict:
    azimuth = float(np.deg2rad(azimuth_deg))
    task = TaskSample(
        cube_xy=(radius_m * float(np.cos(azimuth)), radius_m * float(np.sin(azimuth))),
        cube_yaw=float(np.deg2rad(yaw_deg)),
        box_xy=SCENE_BOX_XY,
        seed=0,
    )
    report = check_task_reachable(task, profile_name)
    row = {
        "radius_m": radius_m,
        "azimuth_deg": azimuth_deg,
        "yaw_deg": yaw_deg,
        "cube_xy": list(task.cube_xy),
        "reachable": report.reachable,
        "reachability_error_m": report.max_error_m,
        "reachability_phase": report.failing_phase,
        "success": False,
        "failure_phase": None,
        "failure_reason": None,
        "discarded": not report.reachable,
    }
    if not report.reachable:
        row["failure_phase"] = f"可达性预检:{report.failing_phase}"
        row["failure_reason"] = f"IK 误差 {report.max_error_m * 1000:.2f} mm 超过该段上限"
        return row

    buffer = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buffer):
            result = run(with_viewer=False, profile_name=profile_name, task=task)
    except Exception as error:  # noqa: BLE001 - 采集/标定需要把任何异常转成一行结果
        result = RunResult(False, False, False, False, "未知", f"{type(error).__name__}: {error}")
    log = buffer.getvalue()
    if verbose:
        print(log)
    elif not result.success:
        print("\n".join(log.splitlines()[-12:]))
    row["duration_sec"] = time.perf_counter() - started
    row["success"] = bool(result.success)
    row["grasp_ok"] = bool(result.grasp_ok)
    row["placed"] = bool(result.placed)
    row["returned"] = bool(result.returned)
    row["failure_phase"] = result.failure_phase
    row["failure_reason"] = result.failure_reason
    return row


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    reachable = [row for row in rows if row["reachable"]]
    successes = [row for row in rows if row["success"]]
    by_radius: dict[str, dict] = {}
    for row in rows:
        key = f"{row['radius_m']:.2f}"
        bucket = by_radius.setdefault(key, {"total": 0, "success": 0, "discarded": 0})
        bucket["total"] += 1
        bucket["success"] += int(row["success"])
        bucket["discarded"] += int(row["discarded"])
    failure_phases: dict[str, int] = {}
    for row in rows:
        if not row["success"]:
            phase = row["failure_phase"] or "未知"
            failure_phases[phase] = failure_phases.get(phase, 0) + 1
    return {
        "grid_cells": total,
        "reachable_cells": len(reachable),
        "discarded_cells": total - len(reachable),
        "successes": len(successes),
        "success_rate": len(successes) / total if total else 0.0,
        "success_rate_among_reachable": len(successes) / len(reachable) if reachable else 0.0,
        "by_radius": by_radius,
        "failure_phases": failure_phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("simulation_fast", "real_safe"), default="real_safe")
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII_M))
    parser.add_argument("--azimuths", type=float, nargs="+", default=list(DEFAULT_AZIMUTHS_DEG))
    parser.add_argument("--yaws", type=float, nargs="+", default=list(DEFAULT_YAWS_DEG))
    parser.add_argument("--output", type=Path, default=Path("results/workspace_calibration.json"))
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--max-discard-rate", type=float, default=0.30)
    parser.add_argument("--verbose", action="store_true", help="打印每次演示的完整日志")
    args = parser.parse_args()

    rows = []
    started = time.time()
    for radius, azimuth, yaw in itertools.product(args.radii, args.azimuths, args.yaws):
        row = probe_cell(radius, azimuth, yaw, args.profile, args.verbose)
        rows.append(row)
        status = "成功" if row["success"] else ("丢弃" if row["discarded"] else "失败")
        print(
            f"r={radius:.3f} θ={azimuth:+6.1f}° yaw={yaw:5.1f}° → {status}"
            f"（预检 {row['reachability_error_m'] * 1000:5.2f} mm"
            f"{'，' + str(row['failure_phase']) if row['failure_phase'] else ''}）",
            flush=True,
        )

    summary = summarise(rows)
    report = {
        "profile": args.profile,
        "grid": {
            "radii_m": list(args.radii),
            "azimuths_deg": list(args.azimuths),
            "yaws_deg": list(args.yaws),
        },
        "summary": summary,
        "cells": rows,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "duration_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n按半径统计：")
    for key, bucket in sorted(summary["by_radius"].items()):
        print(f"  r={key} m: 成功 {bucket['success']}/{bucket['total']}，丢弃 {bucket['discarded']}")
    print("\n失败阶段统计：", summary["failure_phases"] or "无")
    discard_rate = summary["discarded_cells"] / summary["grid_cells"] if summary["grid_cells"] else 0.0
    print(
        f"\n总成功率 {summary['successes']}/{summary['grid_cells']} = "
        f"{summary['success_rate'] * 100:.1f}%；执行过的样本内 "
        f"{summary['successes']}/{summary['reachable_cells']} = "
        f"{summary['success_rate_among_reachable'] * 100:.1f}%；"
        f"执行前丢弃 {summary['discarded_cells']} 个（{discard_rate * 100:.1f}%）"
    )
    print(f"报告写入 {args.output}，耗时 {report['duration_sec']:.1f}s")
    # 判定看「执行过的样本内」成功率：被预检丢弃的样本根本没执行，不算专家失败；
    # 但丢弃率过高说明采样范围有大量无效区域，单独告警。
    if summary["success_rate_among_reachable"] < args.min_success_rate:
        print(f"专家成功率低于 {args.min_success_rate * 100:.0f}%，需要缩小采样范围")
        return 1
    if discard_rate > args.max_discard_rate:
        print(f"丢弃率 {discard_rate * 100:.1f}% 高于 {args.max_discard_rate * 100:.0f}%，建议缩小采样范围")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
