"""相机内参加载与投影的测试。

关键一条是 ost.yaml 与 intrinsics.yaml 的一致性：两份文件手工维护，
数值必须相同，否则 usb_cam（读 ost.yaml）和本工作区（读 intrinsics.yaml）
会得到不同的内参，而且不会报错。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so_ws.common.camera_model import (
    CONFIG_DIR,
    CameraModel,
    load_camera,
    load_camera_from_yaml,
    load_from_ros_ost,
)

CAM_DIR = CONFIG_DIR / "hf868_135_640x480"


def test_hf868_135_loads_expected_values() -> None:
    cam = load_camera("hf868_135_640x480")
    assert (cam.width, cam.height) == (640, 480)
    assert cam.fx == pytest.approx(326.68129, abs=1e-5)
    assert cam.fy == pytest.approx(326.27539, abs=1e-5)
    assert cam.cx == pytest.approx(319.20034, abs=1e-5)
    assert cam.cy == pytest.approx(248.66437, abs=1e-5)
    assert cam.D == pytest.approx([-0.012865, -0.018761, 0.002971, -0.001267, 0.0], abs=1e-6)


def test_yaml_and_ost_agree() -> None:
    """标注版 yaml 与 ROS 原始 ost.yaml 必须给出同一组内参。"""
    from_yaml = load_camera_from_yaml(CAM_DIR / "intrinsics.yaml")
    from_ost = load_from_ros_ost(CAM_DIR / "ost.yaml")

    assert (from_yaml.width, from_yaml.height) == (from_ost.width, from_ost.height)
    assert from_yaml.fx == pytest.approx(from_ost.fx, abs=1e-6)
    assert from_yaml.fy == pytest.approx(from_ost.fy, abs=1e-6)
    assert from_yaml.cx == pytest.approx(from_ost.cx, abs=1e-6)
    assert from_yaml.cy == pytest.approx(from_ost.cy, abs=1e-6)
    assert from_yaml.D == pytest.approx(from_ost.D, abs=1e-6)


def test_hfov_matches_wide_angle_lens() -> None:
    """135° 广角镜头在 640 宽下水平视场应接近 88.8°。"""
    cam = load_camera("hf868_135_640x480")
    assert cam.hfov_deg == pytest.approx(88.8, abs=0.2)
    assert cam.vfov_deg == pytest.approx(72.7, abs=0.2)


def test_project_unproject_roundtrip() -> None:
    """投影再反投影（给定同一深度）应回到原点。"""
    cam = load_camera("hf868_135_640x480")
    rng = np.random.default_rng(0)
    pts = np.stack([
        rng.uniform(-0.3, 0.3, 50),
        rng.uniform(-0.2, 0.2, 50),
        rng.uniform(0.4, 1.2, 50),
    ], axis=-1)

    pixels = cam.project(pts)
    back = cam.unproject(pixels, pts[:, 2])
    assert np.allclose(back, pts, atol=1e-6)


def test_distortion_is_reversible() -> None:
    cam = load_camera("hf868_135_640x480")
    rng = np.random.default_rng(1)
    xy = np.stack([rng.uniform(-0.8, 0.8, 200), rng.uniform(-0.6, 0.6, 200)], axis=-1)

    roundtrip = cam.undistort_normalized(cam.distort_normalized(xy))
    assert np.allclose(roundtrip, xy, atol=1e-9)


def test_zero_distortion_model_is_linear() -> None:
    """无畸变时，投影必须严格等于针孔线性投影。"""
    cam = CameraModel(name="pinhole", width=640, height=480,
                      fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 0.5], [-0.2, -0.1, 2.0]])
    expected = cam.project_ideal(pts)
    assert np.allclose(cam.project(pts), expected, atol=1e-9)


def test_principal_point_projects_to_center_of_optical_axis() -> None:
    cam = load_camera("hf868_135_640x480")
    pixel = cam.project(np.array([[0.0, 0.0, 1.0]]))
    assert pixel[0] == pytest.approx([cam.cx, cam.cy], abs=1e-9)


def test_projection_rejects_points_behind_camera() -> None:
    cam = load_camera("hf868_135_640x480")
    with pytest.raises(ValueError, match="相机前方"):
        cam.project(np.array([[0.0, 0.0, -0.5]]))


def test_pixel_ray_is_unit_length_and_consistent() -> None:
    cam = load_camera("hf868_135_640x480")
    for pixel in ([100.0, 100.0], [320.0, 240.0], [600.0, 400.0]):
        ray = cam.pixel_ray(np.array(pixel))
        assert np.linalg.norm(ray) == pytest.approx(1.0, abs=1e-12)
        # 沿该射线取一点，投影回去应落在原像素
        assert cam.project(ray[None, :] * 0.8)[0] == pytest.approx(pixel, abs=1e-6)


def test_mujoco_convention_conversion_is_its_own_inverse() -> None:
    pts = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
    back = CameraModel.mujoco_to_cam(CameraModel.cam_to_mujoco(pts))
    assert np.allclose(back, pts)
    # MuJoCo 里 -z 才是前方，换算后应变为 +z
    assert CameraModel.cam_to_mujoco(np.array([[0.0, 0.0, 1.0]]))[0, 2] == -1.0


def test_ost_file_exists_in_ros_format() -> None:
    """ost.yaml 要能直接喂给 usb_cam 的 camera_info_url。"""
    path = Path(CAM_DIR / "ost.yaml")
    assert path.is_file()
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("image_width", "image_height", "camera_matrix",
                "distortion_model", "distortion_coefficients",
                "rectification_matrix", "projection_matrix"):
        assert key in data, f"ost.yaml 缺少 {key}"
    assert data["camera_matrix"]["rows"] == 3
    assert data["camera_matrix"]["cols"] == 3
    assert data["distortion_coefficients"]["cols"] == 5


def test_so_arm_ws_takes_priority_over_repo_copy(tmp_path, monkeypatch) -> None:
    """显式指定 so_arm_ws 时，应优先用那里的标定，而不是仓库副本。"""
    arm_ws = tmp_path / "arm_ws"
    cam_dir = arm_ws / "calibration" / "cameras" / "hf868_135_640x480"
    cam_dir.mkdir(parents=True)

    # 造一份 fx 明显不同的标定，用来确认读到的是它
    (cam_dir / "intrinsics.yaml").write_text(
        "resolution: {width: 640, height: 480}\n"
        "intrinsics: {fx: 999.0, fy: 999.0, cx: 320.0, cy: 240.0}\n"
        "distortion: {k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0, k3: 0.0}\n"
        "model: {name: stub}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SO_ARM_WS", str(arm_ws))
    cam = load_camera("hf868_135_640x480")
    assert cam.fx == 999.0
    assert cam.name == "stub"


def test_missing_camera_reports_searched_paths() -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_camera("does_not_exist_camera")
    msg = str(excinfo.value)
    assert "calibration/cameras" in msg
    assert "SO_ARM_WS" in msg


def test_so_arm_ws_calibration_layout_matches_repo_copy() -> None:
    """so_arm_ws 里若存在同名标定（只在部署机上），必须与仓库副本数值一致。

    两边是手工复制的关系，一旦漂移，仿真侧和真机侧会用同一台相机算出不同结果。
    """
    import os

    arm_ws = os.environ.get("SO_ARM_WS") or str(Path.home() / "so_projects" / "so_arm_ws")
    deployed = Path(arm_ws) / "calibration" / "cameras" / "hf868_135_640x480" / "intrinsics.yaml"
    if not deployed.is_file():
        pytest.skip("本机没有部署 so_arm_ws，跳过一致性检查")

    repo = load_camera_from_yaml(CAM_DIR / "intrinsics.yaml")
    live = load_camera_from_yaml(deployed)
    assert (live.width, live.height) == (repo.width, repo.height)
    assert live.fx == pytest.approx(repo.fx, abs=1e-6)
    assert live.fy == pytest.approx(repo.fy, abs=1e-6)
    assert live.cx == pytest.approx(repo.cx, abs=1e-6)
    assert live.cy == pytest.approx(repo.cy, abs=1e-6)
    assert live.D == pytest.approx(repo.D, abs=1e-9)
