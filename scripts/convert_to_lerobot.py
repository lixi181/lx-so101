"""把 `local_raw_v3` 的原始 episode 转换成 LeRobot v3 数据集。

只转换成功的 episode（`<raw-dir>/episode_*`，跳过 `failed/`），并且只接受
`format == local_raw_v3` 且 `git_dirty == false` 的数据。

单位换算：

- 5 个手臂关节由弧度换算成度；
- 夹爪按仿真关节限位线性映射到 0–100，**0 为闭合**：
  `value = (q - low) / (high - low) * 100`，`low`/`high` 取场景里 `gripper` 关节的限位。

夹爪这一段是**仿真近似，接真机前需要按真机标定复核**（见 README）。

时间戳用 `frame_index / fps`，并断言与原始 `timestamp` 一致（容差 1e-9 s；实测 MuJoCo
时间在 577 帧上累计漂移只有 1.8e-12 s）。

本机不装 LeRobot 也能跑 `--dry-run` 和单元测试：模块导入时不 import lerobot，只有真正
写数据集时才延迟导入。

用法：

    ./.venv/bin/python scripts/convert_to_lerobot.py --dry-run
    ./.venv/bin/python scripts/convert_to_lerobot.py \\
        --raw-dir datasets/raw_v1_pilot \\
        --repo-id local/so101_pickplace_v1 \\
        --root datasets/processed/so101_pickplace_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FORMAT = "local_raw_v3"
FPS = 30
ROBOT_TYPE = "so101_follower"
TASK = "Pick up the cube and place it in the box"

# 与 LeRobot 官方 so101 数据集一致的动作/状态键名。
ARM_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
)
GRIPPER_NAME = "gripper.pos"
STATE_NAMES = (*ARM_JOINT_NAMES, GRIPPER_NAME)

GRIPPER_JOINT = "gripper"
GRIPPER_SCALE_MAX = 100.0

# 原始相机目录名 → LeRobot 图像特征名。真机相机尚未定型，所以沿用仿真的 front/wrist；
# 将来真机若用俯视相机，只需改这张表。
CAMERA_FEATURES = {
    "front_camera": "observation.images.front",
    "wrist_camera": "observation.images.wrist",
}

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
TIMESTAMP_ATOL = 1e-9
STATE_ACTION_DIM = 6


@dataclass(frozen=True)
class Episode:
    """一条通过校验的原始 episode。"""

    path: Path
    index: int
    qpos: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    image_size: tuple[int, int]
    sampling_hz: float
    frame_count: int


class EpisodeRejected(Exception):
    """episode 不满足转换条件；消息里说明原因。"""


def gripper_limits(scene_path: Path = FIXED_SCENE_PATH) -> tuple[float, float]:
    """从场景读取夹爪关节限位，作为 0–100 映射的两端。

    从模型读而不是写死常量：限位改了而映射没改会导致夹爪刻度整体偏移。
    """
    model = load_model(scene_path)[0]
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
    if joint_id < 0:
        raise ValueError(f"场景里找不到关节: {GRIPPER_JOINT}")
    low, high = (float(value) for value in model.jnt_range[joint_id])
    if not low < high:
        raise ValueError(f"夹爪关节限位非法: [{low}, {high}]")
    return low, high


def to_lerobot_units(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """把 (..., 6) 的仿真 qpos/action 换算成 LeRobot 单位。

    前 5 列弧度转度，第 6 列（夹爪）线性映射到 0–100。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != STATE_ACTION_DIM:
        raise ValueError(f"期望最后一维是 {STATE_ACTION_DIM}，实际 {array.shape}")
    low, high = limits
    converted = np.empty_like(array)
    converted[..., :5] = np.rad2deg(array[..., :5])
    converted[..., 5] = (array[..., 5] - low) / (high - low) * GRIPPER_SCALE_MAX
    return converted


def load_episode(path: Path) -> Episode:
    """读取并校验一条原始 episode；不满足条件时抛 EpisodeRejected。"""
    metadata_path = path / "episode.json"
    payload_path = path / "state_action.npz"
    if not metadata_path.is_file() or not payload_path.is_file():
        raise EpisodeRejected("缺少 episode.json 或 state_action.npz")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fmt = metadata.get("format")
    if fmt != EXPECTED_FORMAT:
        raise EpisodeRejected(f"format 是 {fmt!r}，只接受 {EXPECTED_FORMAT!r}")
    if metadata.get("git_dirty"):
        raise EpisodeRejected("git_dirty=true，采集时工作区不干净")
    if not metadata.get("success"):
        raise EpisodeRejected("success=false")

    image_size = tuple(metadata.get("image_size", ()))
    if image_size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise EpisodeRejected(f"image_size 是 {image_size}，期望 {(IMAGE_WIDTH, IMAGE_HEIGHT)}")
    sampling_hz = float(metadata.get("sampling_hz", 0.0))
    if sampling_hz != float(FPS):
        raise EpisodeRejected(f"sampling_hz 是 {sampling_hz}，期望 {FPS}")

    with np.load(payload_path) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        action = np.asarray(payload["action"], dtype=np.float64)
        timestamp = np.asarray(payload["timestamp"], dtype=np.float64)

    frame_count = int(metadata.get("frame_count", len(timestamp)))
    if not (len(qpos) == len(action) == len(timestamp) == frame_count):
        raise EpisodeRejected(
            f"帧数不一致: qpos={len(qpos)} action={len(action)} "
            f"timestamp={len(timestamp)} frame_count={frame_count}"
        )
    if qpos.shape[1] != STATE_ACTION_DIM or action.shape[1] != STATE_ACTION_DIM:
        raise EpisodeRejected(f"关节维度不是 {STATE_ACTION_DIM}: {qpos.shape} / {action.shape}")

    check_timestamps(timestamp)
    for camera in CAMERA_FEATURES:
        frames = sorted((path / camera).glob("*.png"))
        if len(frames) != frame_count:
            raise EpisodeRejected(f"{camera} 有 {len(frames)} 张图，期望 {frame_count}")

    return Episode(
        path=path,
        index=int(metadata.get("episode_index", -1)),
        qpos=qpos,
        action=action,
        timestamp=timestamp,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        sampling_hz=sampling_hz,
        frame_count=frame_count,
    )


def check_timestamps(timestamp: np.ndarray, atol: float = TIMESTAMP_ATOL) -> None:
    """断言原始时间戳与 frame_index / FPS 一致。

    MuJoCo 的 `data.time` 是逐步累加的，所以只能按容差比较；实测 577 帧的累计漂移
    是 1.8e-12 s，远小于 1e-9。
    """
    ideal = np.arange(len(timestamp), dtype=np.float64) / FPS
    deviation = float(np.abs(timestamp - ideal).max()) if len(timestamp) else 0.0
    if deviation > atol:
        raise EpisodeRejected(
            f"时间戳与 frame_index/{FPS} 不一致：最大偏差 {deviation:.3e} s 超过容差 {atol:g} s"
        )


def discover_episodes(
    raw_dir: Path, max_episodes: int | None = None
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """列出待转换的 episode，返回 (接受的路径, [(被拒路径, 原因)])。

    只看 `<raw_dir>/episode_*`；`failed/` 下的失败轨迹不参与转换。

    `max_episodes` 只保留**编号最小的 N 条**成功 episode。按编号升序取前缀，所以
    N=50 / 100 / 200 得到的是层层嵌套的子集，便于做数据量消融。
    """
    accepted: list[tuple[int, Path]] = []
    rejected: list[tuple[Path, str]] = []
    for path in sorted(raw_dir.glob("episode_*")):
        if not path.is_dir():
            continue
        try:
            episode = load_episode(path)
        except EpisodeRejected as error:
            rejected.append((path, str(error)))
        else:
            accepted.append((episode.index, path))
    # 按 episode.json 里记录的编号排序，而不是靠目录名字符串。
    accepted.sort(key=lambda item: item[0])
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError(f"--max-episodes 必须大于 0: {max_episodes}")
        accepted = accepted[:max_episodes]
    return [path for _, path in accepted], rejected


def build_features() -> dict:
    """LeRobot v3 的 features 声明。"""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": list(STATE_NAMES),
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }


def load_frame_image(path: Path) -> np.ndarray:
    """读取一帧 PNG 成 RGB uint8 数组。

    LeRobot 的 `validate_feature_image_or_video` 只接受 ndarray 或 PIL Image，
    **不接受文件路径**；用 `with` + `convert` 保证文件句柄及时关闭 —— 一条 episode 有
    `2 × 577` 张图，靠惰性 PIL 对象会一次打开上千个句柄。
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def import_lerobot():
    """延迟导入 LeRobot，缺依赖时给出可执行的提示。"""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:  # pragma: no cover - 本机没装 lerobot
        raise SystemExit(
            "没有找到 lerobot，无法写数据集。本机按约定不装 LeRobot（见 README）：\n"
            "  - 只想看扫描和校验结果：加 --dry-run\n"
            "  - 真正转换：在装了 LeRobot 的服务器上运行本脚本\n"
            f"原始错误：{error}"
        ) from error
    return LeRobotDataset


def convert(
    raw_dir: Path,
    repo_id: str,
    root: Path,
    features: dict,
    limits: tuple[float, float],
    max_episodes: int | None = None,
) -> int:
    """把 raw_dir 下合规 episode 写成一个 LeRobot 数据集。

    `max_episodes` 只取编号最小的 N 条，用来生成嵌套子集。
    """
    LeRobotDataset = import_lerobot()
    episodes, rejected = discover_episodes(raw_dir, max_episodes)
    for path, reason in rejected:
        print(f"跳过 {path.name}: {reason}")
    if not episodes:
        print("没有可转换的 episode", file=sys.stderr)
        return 1

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        robot_type=ROBOT_TYPE,
        root=root,
    )
    for path in episodes:
        episode = load_episode(path)
        states = to_lerobot_units(episode.qpos, limits)
        actions = to_lerobot_units(episode.action, limits)
        for frame in range(episode.frame_count):
            payload = {
                "observation.state": states[frame].astype(np.float32),
                "action": actions[frame].astype(np.float32),
                "task": TASK,
            }
            for camera, feature in CAMERA_FEATURES.items():
                frame_path = path / camera / f"{frame:06d}.png"
                payload[feature] = load_frame_image(frame_path)
            dataset.add_frame(payload)
        dataset.save_episode()
        print(f"已写入 {path.name}（{episode.frame_count} 帧）")

    # finalize 是必需的：不调用就不会把 footer 元数据写进 parquet，数据集无效。
    dataset.finalize()
    print(f"数据集写到 {root}，共 {len(episodes)} 条 episode")
    return 0


def verify(
    root: Path,
    repo_id: str,
    limits: tuple[float, float],
    raw_dir: Path,
    max_episodes: int | None = None,
) -> int:
    """读回数据集，抽查每条 episode 的第 0 帧和最后一帧与原始数据一致。

    state/action 按换算后比较（容差 1e-4），并检查图像形状。
    """
    LeRobotDataset = import_lerobot()
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    # 必须与 convert() 用同一个子集，否则会拿错误的 episode 去比对。
    episodes, _ = discover_episodes(raw_dir, max_episodes)
    print(f"数据集共 {len(dataset)} 帧，原始 {len(episodes)} 条 episode")
    if len(dataset.meta.episodes) != len(episodes):
        print(
            f"数据集里的 episode 数与原始不一致：{len(dataset.meta.episodes)} vs {len(episodes)}",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for position, path in enumerate(episodes):
        episode = load_episode(path)
        # v3 的 episode 边界存在 meta.episodes 的 dataset_from_index（没有 episode_data_index）。
        from_index = int(dataset.meta.episodes[position]["dataset_from_index"])
        indices = [0, episode.frame_count - 1]
        for frame in indices:
            item = dataset[from_index + frame]
            state = np.asarray(item["observation.state"], dtype=np.float64)
            action = np.asarray(item["action"], dtype=np.float64)
            expected_state = to_lerobot_units(episode.qpos[frame], limits)
            expected_action = to_lerobot_units(episode.action[frame], limits)
            for label, got, want in (
                ("state", state, expected_state),
                ("action", action, expected_action),
            ):
                if not np.allclose(got, want, atol=1e-4):
                    print(
                        f"{path.name} 第 {frame} 帧 {label} 不一致：最大偏差 {np.abs(got - want).max():.3e}"
                    )
                    failures += 1
            for camera, feature in CAMERA_FEATURES.items():
                image = np.asarray(item[feature])
                expected_shape = (3, IMAGE_HEIGHT, IMAGE_WIDTH)
                if image.shape not in (expected_shape, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
                    print(f"{path.name} 第 {frame} 帧 {camera} 形状异常: {image.shape}")
                    failures += 1
        print(f"抽查 {path.name}: 第 0 帧与第 {episode.frame_count - 1} 帧")

    if failures:
        print(f"读回校验失败 {failures} 处", file=sys.stderr)
        return 1
    print("读回校验通过")
    return 0


def report(raw_dir: Path, max_episodes: int | None = None) -> int:
    """只扫描并打印统计，不需要 LeRobot。"""
    episodes, rejected = discover_episodes(raw_dir, max_episodes)
    print(f"扫描 {raw_dir}")
    print(f"  接受 {len(episodes)} 条，拒绝 {len(rejected)} 条")
    if max_episodes is not None:
        print(
            f"  只取编号最小的 {len(episodes)} 条（--max-episodes {max_episodes}）："
            f"{episodes[0].name} … {episodes[-1].name}"
        )
    for path, reason in rejected:
        print(f"    拒绝 {path.name}: {reason}")
    if not episodes:
        return 1
    limits = gripper_limits()
    frames = 0
    for path in episodes:
        episode = load_episode(path)
        frames += episode.frame_count
        states = to_lerobot_units(episode.qpos, limits)
        actions = to_lerobot_units(episode.action, limits)
        print(
            f"  {path.name}: {episode.frame_count} 帧，"
            f"state 范围 [{states.min():.2f}, {states.max():.2f}]，"
            f"夹爪 state [{states[:, 5].min():.2f}, {states[:, 5].max():.2f}]，"
            f"夹爪 action [{actions[:, 5].min():.2f}, {actions[:, 5].max():.2f}]"
        )
    print(f"  合计 {frames} 帧，按 {FPS} fps 约 {frames / FPS / 60:.1f} 分钟")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw_v1_pilot"))
    parser.add_argument("--repo-id", default="local/so101_pickplace_v1")
    parser.add_argument("--root", type=Path, default=Path("datasets/processed/so101_pickplace_v1"))
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="只取编号最小的 N 条成功 episode；用于生成 50/100/200 的嵌套子集",
    )
    parser.add_argument("--dry-run", action="store_true", help="只扫描和校验，不写数据集（不需要 LeRobot）")
    parser.add_argument("--verify", action="store_true", help="写完读回校验")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        print("--max-episodes 必须大于 0", file=sys.stderr)
        return 2
    if args.dry_run:
        return report(args.raw_dir, args.max_episodes)
    limits = gripper_limits()
    status = convert(args.raw_dir, args.repo_id, args.root, build_features(), limits, args.max_episodes)
    if status != 0 or not args.verify:
        return status
    return verify(args.root, args.repo_id, limits, args.raw_dir, args.max_episodes)


if __name__ == "__main__":
    raise SystemExit(main())
