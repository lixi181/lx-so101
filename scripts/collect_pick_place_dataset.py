"""采集本地 SO-101 专家轨迹，不依赖 LeRobot。

一个 episode 的写入流程是「先写临时目录，再原子改名」：
成功放到 `raw/episode_XXXXXX/`，失败放到 `raw/failed/episode_XXXXXX/`，两边数据都完整，
失败的可以留作负样本分析。所有 episode 追加一行到 `raw/index.jsonl`，批次信息写
`raw/manifest.json`。

种子由编号推导（`derive_seed`），因此同一个编号永远采样到同一个任务，中途 Ctrl-C
之后续采不会出现重复种子；不指定 `--start-index` 时自动从已有最大编号 +1 开始。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pick_cube
from pick_cube import RunResult, check_task_reachable, run
from PIL import Image

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH
from task_sampling import (
    SAMPLER_MODES,
    SamplerConfig,
    TaskSample,
    cell_for_index,
    derive_seed,
    sample_task,
)

FORMAT_VERSION = "local_raw_v3"
ACTION_SEMANTICS = "absolute_joint_position_target, applied over next tick with linear interpolation"
OBS_ACTION_ALIGNMENT = "obs[t] recorded before action[t] is applied"
DEFAULT_PROFILE = "real_safe"
# 采集默认用分层采样：r/θ/yaw 各 4 档共 64 格，按编号轮转，保证前 N 条覆盖均衡。
DEFAULT_SAMPLER_MODE = "stratified"
TMP_PREFIX = ".tmp_episode_"
FAILED_DIRNAME = "failed"
INDEX_FILENAME = "index.jsonl"
MANIFEST_FILENAME = "manifest.json"
MAX_SAMPLE_ATTEMPTS = 16
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# 环境与版本信息
# --------------------------------------------------------------------------------------


def scene_digest(scene_path: Path) -> dict:
    """计算场景及其 include 的 XML 哈希，用来反查模型版本。"""
    hashes: dict[str, str] = {}

    def visit(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key in hashes:
            return
        payload = resolved.read_bytes()
        hashes[key] = hashlib.sha256(payload).hexdigest()
        for include in re.findall(r'<include\s+file="([^"]+)"', payload.decode("utf-8")):
            visit((resolved.parent / include).resolve())

    visit(scene_path)
    combined = hashlib.sha256()
    for key in sorted(hashes):
        combined.update(Path(key).name.encode("utf-8"))
        combined.update(hashes[key].encode("utf-8"))
    return {
        "sha256": combined.hexdigest(),
        "scene": scene_path.name,
        "files": {Path(key).name: value for key, value in sorted(hashes.items())},
    }


def environment_metadata() -> dict:
    return {
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }


def calibration_metadata(report_path: Path | None) -> dict:
    """读取工作空间标定报告并记录其可复现摘要。"""
    if report_path is None:
        return {}
    resolved = report_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到工作空间标定报告: {report_path}")
    payload = resolved.read_bytes()
    report = json.loads(payload)
    try:
        display_path = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "workspace_calibration": {
            "path": display_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "profile": report.get("profile"),
            "grid": report.get("grid"),
            "summary": report.get("summary"),
        }
    }


def git_metadata() -> dict:
    """返回 git_commit / git_dirty；不在 git 仓库里时视为 dirty。"""

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=False,
                cwd=REPO_ROOT,
            )
        except FileNotFoundError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_commit": commit or "",
        "git_dirty": True if status is None else bool(status),
        "git_status_entries": 0 if status is None else len(status.splitlines()),
    }


# --------------------------------------------------------------------------------------
# 目录与索引
# --------------------------------------------------------------------------------------


def cleanup_tmp_dirs(output_dir: Path) -> list[str]:
    """启动时删掉残留的 .tmp_* 目录。"""
    removed = []
    for path in sorted(output_dir.glob(f"{TMP_PREFIX}*")):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.name)
    return removed


def existing_indices(output_dir: Path) -> list[int]:
    indices = []
    for base in (output_dir, output_dir / FAILED_DIRNAME):
        if not base.is_dir():
            continue
        for path in base.glob("episode_*"):
            suffix = path.name.removeprefix("episode_")
            if path.is_dir() and suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)


def next_index(output_dir: Path) -> int:
    indices = existing_indices(output_dir)
    return indices[-1] + 1 if indices else 0


def append_index_line(output_dir: Path, record: dict) -> None:
    with (output_dir / INDEX_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# 单个 episode
# --------------------------------------------------------------------------------------


def write_episode_payload(tmp_dir: Path, metadata: dict, arrays: dict) -> None:
    np.savez_compressed(tmp_dir / "state_action.npz", **arrays)
    (tmp_dir / "episode.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def make_renderer(model: mujoco.MjModel, height: int, width: int) -> mujoco.Renderer:
    """创建离屏渲染器。单独成函数是为了让测试替换掉它，不依赖 OpenGL 上下文。"""
    return mujoco.Renderer(model, height=height, width=width)


def collect_episode(
    output_dir: Path,
    episode_index: int,
    task: TaskSample,
    width: int,
    height: int,
    profile_name: str,
    base_metadata: dict,
    rejected_samples: int,
    verbose: bool,
) -> tuple[RunResult, Path]:
    """采集一个 episode；先写 .tmp 目录，完成后原子改名到最终位置。"""
    success_path = output_dir / f"episode_{episode_index:06d}"
    failed_path = output_dir / FAILED_DIRNAME / f"episode_{episode_index:06d}"
    for candidate in (success_path, failed_path):
        if candidate.exists():
            raise FileExistsError(f"episode 已存在，为避免覆盖数据而停止: {candidate}")

    tmp_dir = output_dir / f"{TMP_PREFIX}{episode_index:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    (tmp_dir / "front_camera").mkdir(parents=True)
    (tmp_dir / "wrist_camera").mkdir()

    times: list[float] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    cube_pos: list[np.ndarray] = []
    cube_quat: list[np.ndarray] = []
    phases: list[str] = []
    renderer: mujoco.Renderer | None = None

    def observe(model: mujoco.MjModel, data: mujoco.MjData, phase: str, action: np.ndarray) -> None:
        nonlocal renderer
        if renderer is None:
            renderer = make_renderer(model, height, width)
        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pick_cube.CUBE_BODY)
        renderer.update_scene(data, camera="front_camera")
        front = renderer.render().copy()
        renderer.update_scene(data, camera="wrist_camera")
        wrist = renderer.render().copy()
        frame_name = f"{len(times):06d}.png"
        Image.fromarray(front).save(tmp_dir / "front_camera" / frame_name)
        Image.fromarray(wrist).save(tmp_dir / "wrist_camera" / frame_name)
        times.append(float(data.time))
        qpos.append(data.qpos[:6].copy())
        qvel.append(data.qvel[:6].copy())
        actions.append(action[:6].copy())
        cube_pos.append(data.xpos[cube_id].copy())
        cube_quat.append(data.xquat[cube_id].copy())
        phases.append(phase)

    started = time.time()
    # 非 verbose 时吞掉 run() 的逐段日志，避免 100 个 episode 刷屏。
    log_context = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    try:
        with log_context:
            result = run(with_viewer=False, observer=observe, profile_name=profile_name, task=task)
    except (RuntimeError, ValueError) as error:
        # run() 已经把 IK 失败转成 RunResult；这里兜住场景/采样层面的异常。
        result = RunResult(
            success=False,
            grasp_ok=False,
            placed=False,
            returned=False,
            failure_phase=pick_cube.CURRENT_PHASE,
            failure_reason=f"{type(error).__name__}: {error}",
        )
    except BaseException:
        # Ctrl-C 等：不留残缺目录。
        if renderer is not None:
            renderer.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if renderer is not None:
            renderer.close()

    metadata = {
        **base_metadata,
        "episode_index": episode_index,
        "seed": task.seed,
        "success": bool(result.success),
        "grasp_ok": bool(result.grasp_ok),
        "placed": bool(result.placed),
        "returned": bool(result.returned),
        "failure_phase": result.failure_phase,
        "failure_reason": result.failure_reason,
        "frame_count": len(times),
        "image_size": [width, height],
        "cameras": ["front_camera", "wrist_camera"],
        "duration_sec": time.time() - started,
        "lerobot_export": "not_exported",
        "keyframe_reset_used": True,
        "sampler": {
            **base_metadata["sampler"],
            "sample": dataclasses.asdict(task),
            "rejected_samples": rejected_samples,
        },
    }
    write_episode_payload(
        tmp_dir,
        metadata,
        {
            "timestamp": np.asarray(times, dtype=np.float64),
            "qpos": np.asarray(qpos, dtype=np.float64),
            "qvel": np.asarray(qvel, dtype=np.float64),
            "action": np.asarray(actions, dtype=np.float64),
            "cube_pos": np.asarray(cube_pos, dtype=np.float64),
            "cube_quat": np.asarray(cube_quat, dtype=np.float64),
            "phase": np.asarray(phases),
        },
    )

    target = success_path if result.success else failed_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(target)  # 同一文件系统内的 rename 是原子操作
    return result, target


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="采集 episode 数量")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="episode 编号起点；省略时自动从已有最大编号 +1 开始",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--seed", type=int, default=20261010, help="基础种子，最终种子由编号推导")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--profile", choices=("real_safe", "simulation_fast"), default=DEFAULT_PROFILE)
    parser.add_argument("--radius-min", type=float, default=SamplerConfig.radius_m[0])
    parser.add_argument("--radius-max", type=float, default=SamplerConfig.radius_m[1])
    parser.add_argument("--azimuth-min", type=float, default=SamplerConfig.azimuth_deg[0])
    parser.add_argument("--azimuth-max", type=float, default=SamplerConfig.azimuth_deg[1])
    parser.add_argument("--yaw-min", type=float, default=SamplerConfig.cube_yaw_deg[0])
    parser.add_argument("--yaw-max", type=float, default=SamplerConfig.cube_yaw_deg[1])
    parser.add_argument("--box-jitter", type=float, default=SamplerConfig.box_jitter_m)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        help="工作空间标定 JSON；其路径、哈希和摘要会写进 manifest/episode 元数据",
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=None,
        help="独立评估布局的基础 seed；仅写入批次元数据，不参与训练采样",
    )
    parser.add_argument(
        "--sampler-mode",
        choices=SAMPLER_MODES,
        default=DEFAULT_SAMPLER_MODE,
        help="stratified（默认）把 r/θ/yaw 各分 --bins 档后按编号轮转；uniform 完全均匀随机",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs=3,
        metavar=("R", "THETA", "YAW"),
        default=list(SamplerConfig.bins),
        help="分层档数，默认 4 4 4（共 64 格）",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="工作区有未提交改动时仍然采集，并在元数据里标记 git_dirty",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每次演示的完整日志")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.episodes < 1:
        print("--episodes 必须大于 0", file=sys.stderr)
        return 2
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    git = git_metadata()
    if git["git_dirty"] and not args.allow_dirty:
        print(
            "工作区有未提交的改动，拒绝采集（用 --allow-dirty 跳过，但会在元数据里标记 "
            f"git_dirty）。git status 条目数: {git['git_status_entries']}",
            file=sys.stderr,
        )
        return 2

    removed = cleanup_tmp_dirs(output_dir)
    if removed:
        print(f"清理残留临时目录: {', '.join(removed)}")

    scene = scene_digest(FIXED_SCENE_PATH)
    model = mujoco.MjModel.from_xml_path(str(FIXED_SCENE_PATH))
    timestep = float(model.opt.timestep)
    profile = pick_cube.get_motion_profile(args.profile)
    # 控制周期由 physics_per_tick × timestep 计算，不再写死。
    control_period = profile.physics_per_tick * timestep
    sampler_config = SamplerConfig(
        radius_m=(args.radius_min, args.radius_max),
        azimuth_deg=(args.azimuth_min, args.azimuth_max),
        cube_yaw_deg=(args.yaw_min, args.yaw_max),
        box_jitter_m=args.box_jitter,
        mode=args.sampler_mode,
        bins=tuple(args.bins),
    )
    base_metadata = {
        "format": FORMAT_VERSION,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        **environment_metadata(),
        "scene": f"models/so101/{FIXED_SCENE_PATH.name}",
        "scene_sha256": scene["sha256"],
        "scene_files": scene["files"],
        "motion_profile": dataclasses.asdict(profile),
        "control_period_sec": control_period,
        "sampling_hz": 1.0 / control_period,
        "control_source": "scripts/pick_cube.py",
        "action_semantics": ACTION_SEMANTICS,
        "obs_action_alignment": OBS_ACTION_ALIGNMENT,
        # sampler_mode 与格子编号一起记录，便于事后核对覆盖与复现任务。
        "sampler_mode": sampler_config.mode,
        "sampler": dataclasses.asdict(sampler_config),
        **calibration_metadata(args.calibration_report),
        "evaluation_base_seed": args.eval_base_seed,
    }

    start_index = args.start_index if args.start_index is not None else next_index(output_dir)
    if start_index < 0:
        print("--start-index 不能为负", file=sys.stderr)
        return 2

    print(
        f"开始采集: 编号 {start_index} 起共 {args.episodes} 个，profile={args.profile}，"
        f"git={git['git_commit'][:8] or 'unknown'}{'(dirty)' if git['git_dirty'] else ''}，"
        f"控制周期 {control_period:.4f}s"
    )
    print(f"采样范围: {dataclasses.asdict(sampler_config)}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    started_clock = time.time()
    successes = 0
    failures = 0
    discarded_samples = 0
    discarded_episodes = 0
    failure_phases: dict[str, int] = {}
    episode_records: list[dict] = []

    try:
        for offset in range(args.episodes):
            index = start_index + offset
            # 分层采样：格子只由编号决定，重采样换种子但不换格子。
            cell = cell_for_index(index, sampler_config) if sampler_config.mode == "stratified" else None
            task = None
            rejected = 0
            for attempt in range(MAX_SAMPLE_ATTEMPTS):
                candidate = sample_task(derive_seed(args.seed, index, attempt), sampler_config, cell=cell)
                report = check_task_reachable(candidate, args.profile)
                if report.reachable:
                    task = candidate
                    break
                rejected += 1
                print(
                    f"  丢弃样本 episode_{index:06d} attempt={attempt}: "
                    f"{report.failing_phase}（IK {report.max_error_m * 1000:.2f} mm）"
                )
            discarded_samples += rejected
            if task is None:
                discarded_episodes += 1
                print(f"episode_{index:06d}: {MAX_SAMPLE_ATTEMPTS} 次采样都不可达，跳过该编号")
                append_index_line(
                    output_dir,
                    {
                        "episode_index": index,
                        "format": FORMAT_VERSION,
                        "success": False,
                        "discarded": True,
                        "rejected_samples": rejected,
                        "failure_phase": "采样",
                        "failure_reason": f"{MAX_SAMPLE_ATTEMPTS} 次采样都不可达",
                        "git_commit": git["git_commit"],
                        "git_dirty": git["git_dirty"],
                    },
                )
                continue

            print(
                f"采集 episode_{index:06d}: cube_xy=({task.cube_xy[0]:.4f}, {task.cube_xy[1]:.4f}) "
                f"cube_yaw={task.cube_yaw_deg:.2f}° seed={task.seed} "
                f"cell={task.cell if task.cell is not None else '-'} 丢弃样本 {rejected}",
                flush=True,
            )
            result, target = collect_episode(
                output_dir,
                index,
                task,
                args.width,
                args.height,
                args.profile,
                base_metadata,
                rejected,
                args.verbose,
            )

            successes += int(result.success)
            failures += int(not result.success)
            if not result.success:
                phase = result.failure_phase or "未知"
                failure_phases[phase] = failure_phases.get(phase, 0) + 1
            record = {
                "episode_index": index,
                "format": FORMAT_VERSION,
                "seed": task.seed,
                "success": bool(result.success),
                "discarded": False,
                "grasp_ok": bool(result.grasp_ok),
                "placed": bool(result.placed),
                "returned": bool(result.returned),
                "failure_phase": result.failure_phase,
                "failure_reason": result.failure_reason,
                "path": str(target.relative_to(output_dir)),
                "cube_xy": list(task.cube_xy),
                "cube_yaw_deg": task.cube_yaw_deg,
                "rejected_samples": rejected,
                "git_commit": git["git_commit"],
                "git_dirty": git["git_dirty"],
            }
            append_index_line(output_dir, record)
            episode_records.append(record)
            status = "成功" if result.success else f"失败（{result.failure_phase}）"
            print(f"episode_{index:06d}: {status} → {target.relative_to(output_dir)}", flush=True)
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，已完成的 episode 保持完整；残留临时目录会在下次启动时清理。", file=sys.stderr)
        _write_manifest(
            output_dir,
            args,
            sampler_config,
            base_metadata,
            started_at,
            started_clock,
            {
                "requested": args.episodes,
                "completed": len(episode_records),
                "successes": successes,
                "failures": failures,
                "discarded_samples": discarded_samples,
                "discarded_episodes": discarded_episodes,
                "failure_phases": failure_phases,
                "interrupted": True,
            },
            episode_records,
        )
        return 130

    completed = successes + failures
    success_rate = successes / completed if completed else 0.0
    summary = {
        "requested": args.episodes,
        "completed": completed,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "discarded_samples": discarded_samples,
        "discarded_episodes": discarded_episodes,
        "failure_phases": failure_phases,
        "interrupted": False,
    }
    manifest_path = _write_manifest(
        output_dir,
        args,
        sampler_config,
        base_metadata,
        started_at,
        started_clock,
        summary,
        episode_records,
    )

    print("\n采集汇总:")
    print(f"  完成 {completed}/{args.episodes}，成功 {successes}，失败 {failures}")
    print(f"  成功率 {success_rate * 100:.1f}%（阈值 {args.min_success_rate * 100:.0f}%）")
    print(f"  按失败阶段统计: {failure_phases or '无'}")
    print(f"  因不可达丢弃的样本数: {discarded_samples}，整编号跳过的: {discarded_episodes}")
    print(f"  manifest: {manifest_path}")
    return 0 if success_rate >= args.min_success_rate else 1


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    sampler_config: SamplerConfig,
    base_metadata: dict,
    started_at: str,
    started_clock: float,
    summary: dict,
    episode_records: list[dict],
) -> Path:
    manifest = {
        **base_metadata,
        "format": FORMAT_VERSION,
        "command_line": sys.argv,
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_sec": time.time() - started_clock,
        "sampler": dataclasses.asdict(sampler_config),
        "summary": summary,
        "episodes": episode_records,
    }
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
