"""验证采集器的稳健性：原子写入、失败归档、续采种子、临时目录清理和元数据。

物理部分用假的 run() 替换，只验证目录、索引和元数据的管道；真实专家演示的行为由
tests/test_pick_cube.py 和 tests/test_action_alignment.py 覆盖。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from module_loader import load_script
from PIL import Image

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model
from task_sampling import SamplerConfig, TaskSample, derive_seed

COLLECTOR_SCRIPT = "scripts/collect_pick_place_dataset.py"


class FakeRenderer:
    """替代 mujoco.Renderer：按尺寸返回黑图，不创建 OpenGL 上下文。

    没有图形会话的环境（SSH、CI、macOS 无 WindowServer 连接）里真 Renderer 会抛
    CGLError / GLError；这些测试只关心目录与元数据，不关心像素。
    """

    def __init__(self, model, height: int, width: int):
        self.height = height
        self.width = width
        self.cameras: list[str] = []
        self.closed = False

    def update_scene(self, data, camera: str) -> None:
        self.cameras.append(camera)

    def render(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def collector(monkeypatch):
    module = load_script(COLLECTOR_SCRIPT, "collect_pick_place_dataset_test")
    monkeypatch.setattr(module, "make_renderer", FakeRenderer)
    return module


def make_task(index: int = 0) -> TaskSample:
    return TaskSample(cube_xy=(0.22, 0.0), cube_yaw=0.1, box_xy=(0.04, 0.18), seed=derive_seed(7, index))


def fake_run_factory(collector, success: bool):
    """返回一个只跑两帧、不推进物理的假 run()。"""

    def fake_run(with_viewer, observer=None, profile_name="real_safe", task=None):
        model, data = load_model(FIXED_SCENE_PATH)
        if observer is not None:
            for _ in range(2):
                observer(model, data, "fake", np.zeros(6))
        return collector.RunResult(
            success=success,
            grasp_ok=success,
            placed=success,
            returned=success,
            failure_phase=None if success else "抬升",
            failure_reason=None if success else "假失败",
        )

    return fake_run


def base_metadata(collector) -> dict:
    scene = collector.scene_digest(FIXED_SCENE_PATH)
    return {
        "format": collector.FORMAT_VERSION,
        "scene_sha256": scene["sha256"],
        "sampler": {"radius_m": [0.20, 0.24], "azimuth_deg": [-30.0, 30.0]},
    }


# --------------------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------------------


def test_cleanup_removes_leftover_tmp_dirs(collector, tmp_path):
    (tmp_path / ".tmp_episode_000004").mkdir()
    (tmp_path / ".tmp_episode_000004" / "episode.json").write_text("{}")
    (tmp_path / ".tmp_episode_000009").mkdir()
    keep = tmp_path / "episode_000000"
    keep.mkdir()

    removed = collector.cleanup_tmp_dirs(tmp_path)

    assert removed == [".tmp_episode_000004", ".tmp_episode_000009"]
    assert not (tmp_path / ".tmp_episode_000004").exists()
    assert keep.exists()


def test_next_index_considers_success_and_failed_dirs(collector, tmp_path):
    assert collector.next_index(tmp_path) == 0
    (tmp_path / "episode_000000").mkdir()
    (tmp_path / "episode_000002").mkdir()
    (tmp_path / collector.FAILED_DIRNAME / "episode_000005").mkdir(parents=True)
    assert collector.existing_indices(tmp_path) == [0, 2, 5]
    assert collector.next_index(tmp_path) == 6


def test_scene_digest_covers_included_xml(collector):
    digest = collector.scene_digest(FIXED_SCENE_PATH)
    assert digest["sha256"] and len(digest["sha256"]) == 64
    assert "scene_fixed.xml" in digest["files"]
    assert "so101_new_calib_camera_render.xml" in digest["files"]
    assert digest == collector.scene_digest(FIXED_SCENE_PATH)


def test_calibration_metadata_records_report_digest(collector, tmp_path):
    report = tmp_path / "calibration.json"
    payload = {"profile": "real_safe", "grid": {"radii_m": [0.2, 0.3]}, "summary": {"successes": 2}}
    report.write_text(json.dumps(payload), encoding="utf-8")

    metadata = collector.calibration_metadata(report)

    assert metadata["workspace_calibration"]["path"] == str(report)
    assert len(metadata["workspace_calibration"]["sha256"]) == 64
    assert metadata["workspace_calibration"]["profile"] == "real_safe"
    assert metadata["workspace_calibration"]["grid"] == payload["grid"]
    assert metadata["workspace_calibration"]["summary"] == payload["summary"]


def test_seeds_are_stable_per_index_and_do_not_repeat(collector):
    """修复后的种子只由编号决定：续采不会重复，也不会因为起始编号不同而改变。"""
    base = 20261010
    first_run = [derive_seed(base, index) for index in range(3)]
    second_run = [derive_seed(base, index) for index in range(3, 6)]
    resumed = [derive_seed(base, index) for index in range(3, 6)]
    assert second_run == resumed
    assert not set(first_run) & set(second_run)
    assert len(set(first_run + second_run)) == 6


# --------------------------------------------------------------------------------------
# 单个 episode 的写入
# --------------------------------------------------------------------------------------


def test_successful_episode_is_renamed_into_place(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    task = make_task()
    result, target = collector.collect_episode(
        tmp_path, 0, task, 16, 12, "real_safe", base_metadata(collector), 0, verbose=False
    )

    assert result.success
    assert target == tmp_path / "episode_000000"
    assert not list(tmp_path.glob(f"{collector.TMP_PREFIX}*")), "不能留下临时目录"
    assert (target / "episode.json").is_file()
    assert (target / "state_action.npz").is_file()
    assert (target / "front_camera" / "000000.png").is_file()
    assert (target / "wrist_camera" / "000001.png").is_file()

    metadata = json.loads((target / "episode.json").read_text())
    assert metadata["format"] == collector.FORMAT_VERSION
    assert metadata["success"] is True
    assert metadata["frame_count"] == 2
    assert metadata["sampler"]["sample"]["seed"] == task.seed
    assert metadata["sampler"]["rejected_samples"] == 0
    with np.load(target / "state_action.npz") as payload:
        assert payload["action"].shape == (2, 6)
        assert payload["qpos"].shape == (2, 6)


def test_renderer_renders_both_cameras_and_is_closed(collector, tmp_path, monkeypatch):
    created: list[FakeRenderer] = []

    def tracking_renderer(model, height, width):
        renderer = FakeRenderer(model, height, width)
        created.append(renderer)
        return renderer

    monkeypatch.setattr(collector, "make_renderer", tracking_renderer)
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    _, target = collector.collect_episode(
        tmp_path, 0, make_task(), 16, 12, "real_safe", base_metadata(collector), 0, verbose=False
    )

    assert len(created) == 1, "一个 episode 只创建一个渲染器"
    assert created[0].closed
    assert created[0].cameras == ["front_camera", "wrist_camera"] * 2
    with Image.open(target / "front_camera" / "000000.png") as frame:
        assert frame.size == (16, 12)


def test_failed_episode_is_archived_under_failed_dir(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, False))
    result, target = collector.collect_episode(
        tmp_path, 3, make_task(3), 16, 12, "real_safe", base_metadata(collector), 2, verbose=False
    )

    assert not result.success
    assert target == tmp_path / collector.FAILED_DIRNAME / "episode_000003"
    assert (target / "episode.json").is_file(), "失败 episode 的数据也要完整保留"
    assert not (tmp_path / "episode_000003").exists()
    assert not list(tmp_path.glob(f"{collector.TMP_PREFIX}*"))
    metadata = json.loads((target / "episode.json").read_text())
    assert metadata["success"] is False
    assert metadata["failure_phase"] == "抬升"
    assert metadata["sampler"]["rejected_samples"] == 2


def test_existing_episode_is_never_overwritten(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    (tmp_path / "episode_000000").mkdir()
    with pytest.raises(FileExistsError):
        collector.collect_episode(
            tmp_path, 0, make_task(), 16, 12, "real_safe", base_metadata(collector), 0, verbose=False
        )


def test_exception_does_not_leave_partial_directory(collector, tmp_path, monkeypatch):
    def exploding_run(with_viewer, observer=None, profile_name="real_safe", task=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(collector, "run", exploding_run)
    with pytest.raises(KeyboardInterrupt):
        collector.collect_episode(
            tmp_path, 0, make_task(), 16, 12, "real_safe", base_metadata(collector), 0, verbose=False
        )
    assert not list(tmp_path.glob(f"{collector.TMP_PREFIX}*")), "异常路径也必须清掉临时目录"
    assert not (tmp_path / "episode_000000").exists()


# --------------------------------------------------------------------------------------
# main() 端到端（采集逻辑用假 run）
# --------------------------------------------------------------------------------------


def run_main(collector, monkeypatch, argv):
    monkeypatch.setattr(collector.sys, "argv", ["collect_pick_place_dataset.py", *argv])
    return collector.main()


def test_main_writes_index_and_manifest(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    exit_code = run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "2",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--width",
            "16",
            "--height",
            "12",
            "--eval-base-seed",
            "20261011",
        ],
    )
    assert exit_code == 0

    lines = (tmp_path / collector.INDEX_FILENAME).read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["episode_index"] for record in records] == [0, 1]
    assert len({record["seed"] for record in records}) == 2, "不同编号不能重复使用同一个种子"
    assert all(record["success"] for record in records)

    manifest = json.loads((tmp_path / collector.MANIFEST_FILENAME).read_text())
    assert manifest["format"] == collector.FORMAT_VERSION
    assert manifest["summary"]["successes"] == 2
    assert manifest["summary"]["interrupted"] is False
    assert manifest["arguments"]["episodes"] == "2"
    assert "git_commit" in manifest and "scene_sha256" in manifest
    assert manifest["evaluation_base_seed"] == 20261011


def test_collector_defaults_to_stratified_and_records_cell(collector, tmp_path, monkeypatch):
    """采集默认走分层采样，并且 episode.json 里要能查到 sampler_mode 和格子编号。"""
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    exit_code = run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "3",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--width",
            "16",
            "--height",
            "12",
        ],
    )
    assert exit_code == 0

    metadata = [
        json.loads((tmp_path / f"episode_{index:06d}" / "episode.json").read_text()) for index in range(3)
    ]
    assert all(item["sampler_mode"] == "stratified" for item in metadata)
    assert all(item["sampler"]["mode"] == "stratified" for item in metadata)
    assert [item["sampler"]["sample"]["cell"] for item in metadata] == [0, 1, 2]

    manifest = json.loads((tmp_path / collector.MANIFEST_FILENAME).read_text())
    assert manifest["sampler_mode"] == "stratified"


def test_collector_can_switch_to_uniform(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    exit_code = run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "1",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--sampler-mode",
            "uniform",
            "--width",
            "16",
            "--height",
            "12",
        ],
    )
    assert exit_code == 0
    metadata = json.loads((tmp_path / "episode_000000" / "episode.json").read_text())
    assert metadata["sampler_mode"] == "uniform"
    assert metadata["sampler"]["sample"]["cell"] is None


def test_stratified_collection_covers_every_cell_exactly_once(collector, tmp_path, monkeypatch):
    """走完一次完整轮转（64 条）时每格恰好 1 条，端到端验证均衡性。"""
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    exit_code = run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "64",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--width",
            "16",
            "--height",
            "12",
        ],
    )
    assert exit_code == 0
    cells = [
        json.loads((tmp_path / f"episode_{index:06d}" / "episode.json").read_text())["sampler"]["sample"][
            "cell"
        ]
        for index in range(64)
    ]
    assert sorted(cells) == list(range(64))


def test_main_resumes_from_existing_index(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    argv = [
        "--episodes",
        "1",
        "--output-dir",
        str(tmp_path),
        "--allow-dirty",
        "--width",
        "16",
        "--height",
        "12",
    ]
    assert run_main(collector, monkeypatch, argv) == 0
    assert run_main(collector, monkeypatch, argv) == 0

    records = [
        json.loads(line) for line in (tmp_path / collector.INDEX_FILENAME).read_text().strip().splitlines()
    ]
    assert [record["episode_index"] for record in records] == [0, 1]
    assert len({record["seed"] for record in records}) == 2


def test_main_reports_low_success_rate_with_nonzero_exit(collector, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, False))
    exit_code = run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "2",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--min-success-rate",
            "0.9",
            "--width",
            "16",
            "--height",
            "12",
        ],
    )
    assert exit_code == 1
    manifest = json.loads((tmp_path / collector.MANIFEST_FILENAME).read_text())
    assert manifest["summary"]["success_rate"] == 0.0
    assert manifest["summary"]["failure_phases"] == {"抬升": 2}
    assert (tmp_path / collector.FAILED_DIRNAME / "episode_000000").is_dir()


def test_main_cleans_tmp_dirs_on_start(collector, tmp_path, monkeypatch):
    stale = tmp_path / f"{collector.TMP_PREFIX}000042"
    stale.mkdir()
    (stale / "episode.json").write_text("{}")
    monkeypatch.setattr(collector, "run", fake_run_factory(collector, True))
    run_main(
        collector,
        monkeypatch,
        [
            "--episodes",
            "1",
            "--output-dir",
            str(tmp_path),
            "--allow-dirty",
            "--width",
            "16",
            "--height",
            "12",
        ],
    )
    assert not stale.exists()


def test_sampler_config_defaults_match_calibration(collector):
    """默认采样范围必须与 README 里标定的结论一致。"""
    config = SamplerConfig()
    assert config.radius_m == (0.20, 0.24)
    assert config.azimuth_deg == (-30.0, 30.0)
    assert config.cube_yaw_deg == (0.0, 90.0)
