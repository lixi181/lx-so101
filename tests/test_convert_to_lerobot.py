"""验证 local_raw_v3 → LeRobot 的转换：过滤规则、单位换算、时间戳断言。

不依赖 LeRobot 的部分覆盖发现/校验/换算；真正写数据集的往返测试在本机没装 LeRobot 时
自动跳过（本机按约定不装，见 README），在服务器上会实际执行。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from module_loader import load_script
from PIL import Image

CONVERTER_SCRIPT = "scripts/convert_to_lerobot.py"
FRAMES = 4


@pytest.fixture
def converter():
    return load_script(CONVERTER_SCRIPT, "convert_to_lerobot_test")


def make_episode(
    root: Path,
    index: int = 0,
    frames: int = FRAMES,
    timestamp: np.ndarray | None = None,
    **metadata_overrides,
) -> Path:
    """造一条结构合法的 v3 episode；metadata_overrides 用来构造各种拒绝场景。"""
    path = root / f"episode_{index:06d}"
    for camera in ("front_camera", "wrist_camera"):
        (path / camera).mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            Image.new("RGB", (320, 240)).save(path / camera / f"{frame:06d}.png")

    metadata = {
        "format": "local_raw_v3",
        "git_dirty": False,
        "success": True,
        "image_size": [320, 240],
        "sampling_hz": 30.0,
        "frame_count": frames,
        "episode_index": index,
    }
    metadata.update(metadata_overrides)
    (path / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")

    qpos = np.zeros((frames, 6), dtype=np.float64)
    qpos[:, 0] = np.linspace(0.0, 1.0, frames)
    qpos[:, 5] = np.linspace(0.0, 1.0, frames)
    np.savez_compressed(
        path / "state_action.npz",
        qpos=qpos,
        action=qpos + 0.5,
        timestamp=np.arange(frames) / 30.0 if timestamp is None else timestamp,
    )
    return path


# --------------------------------------------------------------------------------------
# 单位换算
# --------------------------------------------------------------------------------------


def test_arm_joints_convert_from_radians_to_degrees(converter):
    values = np.zeros((1, 6))
    values[0, :5] = np.deg2rad([90.0, -45.0, 0.0, 30.0, -120.0])
    converted = converter.to_lerobot_units(values, (0.0, 1.0))
    assert converted[0, :5] == pytest.approx([90.0, -45.0, 0.0, 30.0, -120.0])


def test_gripper_maps_joint_limits_to_zero_and_hundred(converter):
    """0 为闭合：关节下限 → 0，上限 → 100，中点是 50。"""
    low, high = -0.1745, 1.7453
    values = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, low], [0, 0, 0, 0, 0, high], [0, 0, 0, 0, 0, (low + high) / 2]]
    )
    converted = converter.to_lerobot_units(values, (low, high))
    assert converted[0, 5] == pytest.approx(0.0, abs=1e-12)
    assert converted[1, 5] == pytest.approx(100.0, abs=1e-12)
    assert converted[2, 5] == pytest.approx(50.0, abs=1e-12)


def test_gripper_limits_come_from_the_scene(converter):
    low, high = converter.gripper_limits()
    assert (low, high) == pytest.approx((-0.1745, 1.7453), abs=1e-4)


def test_conversion_rejects_wrong_joint_count(converter):
    with pytest.raises(ValueError, match="最后一维"):
        converter.to_lerobot_units(np.zeros((3, 5)), (0.0, 1.0))


# --------------------------------------------------------------------------------------
# 时间戳
# --------------------------------------------------------------------------------------


def test_timestamps_at_frame_over_fps_pass(converter):
    converter.check_timestamps(np.arange(100) / 30.0)


def test_timestamps_tolerate_mujoco_accumulation(converter):
    """MuJoCo 的 data.time 是累加的，允许浮点漂移但不允许错帧。"""
    drifted = np.arange(577) / 30.0 + 1.8e-12
    converter.check_timestamps(drifted)


def test_timestamps_reject_a_shifted_frame(converter):
    with pytest.raises(converter.EpisodeRejected, match="时间戳"):
        converter.check_timestamps(np.arange(10) / 30.0 + 0.031)


# --------------------------------------------------------------------------------------
# episode 校验
# --------------------------------------------------------------------------------------


def test_valid_episode_loads(converter, tmp_path):
    path = make_episode(tmp_path)
    episode = converter.load_episode(path)
    assert episode.frame_count == FRAMES
    assert episode.qpos.shape == (FRAMES, 6)
    assert episode.image_size == (320, 240)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"format": "local_raw_v2"}, "format"),
        ({"git_dirty": True}, "git_dirty"),
        ({"success": False}, "success"),
        ({"image_size": [160, 120]}, "image_size"),
        ({"sampling_hz": 31.25}, "sampling_hz"),
        ({"frame_count": FRAMES + 3}, "帧数不一致"),
    ],
)
def test_invalid_episodes_are_rejected(converter, tmp_path, overrides, match):
    path = make_episode(tmp_path, **overrides)
    with pytest.raises(converter.EpisodeRejected, match=match):
        converter.load_episode(path)


def test_missing_camera_frames_are_rejected(converter, tmp_path):
    path = make_episode(tmp_path)
    (path / "front_camera" / "000000.png").unlink()
    with pytest.raises(converter.EpisodeRejected, match="front_camera"):
        converter.load_episode(path)


def test_discover_skips_failed_directory_and_reports_reasons(converter, tmp_path):
    make_episode(tmp_path, index=0)
    make_episode(tmp_path, index=1, format="local_raw_v2")
    failed = tmp_path / "failed" / "episode_000002"
    failed.mkdir(parents=True)
    (failed / "episode.json").write_text(json.dumps({"format": "local_raw_v3"}), encoding="utf-8")

    accepted, rejected = converter.discover_episodes(tmp_path)
    assert [path.name for path in accepted] == ["episode_000000"]
    assert [path.name for path, _ in rejected] == ["episode_000001"]


def test_frame_image_loads_as_rgb_uint8(converter, tmp_path):
    """add_frame 只接受 ndarray / PIL Image，所以脚本自己读图，形状必须是 (H, W, 3)。"""
    path = make_episode(tmp_path)
    image = converter.load_frame_image(path / "front_camera" / "000000.png")
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8


# --------------------------------------------------------------------------------------
# --max-episodes：取编号最小的 N 条成功 episode
# --------------------------------------------------------------------------------------


def test_max_episodes_keeps_the_smallest_indices(converter, tmp_path):
    for index in range(6):
        make_episode(tmp_path, index=index)
    accepted, rejected = converter.discover_episodes(tmp_path, max_episodes=3)
    assert [path.name for path in accepted] == [
        "episode_000000",
        "episode_000001",
        "episode_000002",
    ]
    assert rejected == []


def test_max_episodes_subsets_are_nested(converter, tmp_path):
    """50 / 100 / 200 必须是层层嵌套的子集，否则数据量消融不可比。"""
    for index in range(6):
        make_episode(tmp_path, index=index)
    subsets = {
        count: [path.name for path in converter.discover_episodes(tmp_path, max_episodes=count)[0]]
        for count in (2, 4, 6)
    }
    assert subsets[2] == subsets[4][:2]
    assert subsets[4] == subsets[6][:4]


def test_max_episodes_counts_only_accepted_episodes(converter, tmp_path):
    """被拒的 episode 不占名额：N 是对成功 episode 而言的。"""
    make_episode(tmp_path, index=0)
    make_episode(tmp_path, index=1, format="local_raw_v2")
    make_episode(tmp_path, index=2)
    accepted, rejected = converter.discover_episodes(tmp_path, max_episodes=2)
    assert [path.name for path in accepted] == ["episode_000000", "episode_000002"]
    assert [path.name for path, _ in rejected] == ["episode_000001"]


def test_max_episodes_larger_than_available_is_a_no_op(converter, tmp_path):
    for index in range(3):
        make_episode(tmp_path, index=index)
    accepted, _ = converter.discover_episodes(tmp_path, max_episodes=50)
    assert len(accepted) == 3


def test_max_episodes_rejects_zero(converter, tmp_path):
    make_episode(tmp_path, index=0)
    with pytest.raises(ValueError, match="必须大于 0"):
        converter.discover_episodes(tmp_path, max_episodes=0)


# --------------------------------------------------------------------------------------
# 往返（需要 LeRobot；本机按约定不装时跳过）
# --------------------------------------------------------------------------------------


def test_round_trip_matches_raw_data(converter, tmp_path, max_episodes=None):
    pytest.importorskip("lerobot", reason="本机按约定不装 LeRobot，转换在服务器上执行")
    raw_dir = tmp_path / "raw"
    for index in range(3):
        make_episode(raw_dir, index=index)
    root = tmp_path / "processed"

    limits = converter.gripper_limits()
    status = converter.convert(raw_dir, "local/test", root, converter.build_features(), limits, max_episodes)
    assert status == 0
    # verify 必须和 convert 用同一个子集，否则会拿错误的 episode 比对。
    assert converter.verify(root, "local/test", limits, raw_dir, max_episodes) == 0


def test_round_trip_with_max_episodes_uses_the_same_subset(converter, tmp_path):
    test_round_trip_matches_raw_data(converter, tmp_path, max_episodes=2)
