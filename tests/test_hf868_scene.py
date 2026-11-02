"""守护 scene_hf868.xml：外参场景必须能加载，且与原场景不冲突。

这个文件的相机参数是 provisional 的（详见 README「相机外参」），但结构必须稳定：
一旦 include 写错、xyaxes 写错或原场景的相机被覆盖，这里就会失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models/so101/scene_hf868.xml"
EXTRINSICS = ROOT / "results/extrinsics_hf868.json"


def _camera_id(model: mujoco.MjModel, name: str) -> int:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert cid >= 0, f"场景里没有相机 {name!r}"
    return cid


def test_scene_loads_and_keeps_original_cameras() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # 原场景的相机必须还在（叠加不能把它们覆盖掉）
    for name in ("front_camera", "wrist_camera"):
        _camera_id(model, name)
    _camera_id(model, "hf868_camera")
    assert model.ncam == 3
    # 原场景的物体也还在
    for body in ("cube", "target_box", "wrist", "gripper"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready") >= 0


def test_camera_pose_matches_extrinsics_json() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    cid = _camera_id(model, "hf868_camera")
    expected = json.loads(EXTRINSICS.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        model.cam_pos[cid], expected["camera_pos_m"], atol=1e-6
    )


def test_focal_length_is_honoured() -> None:
    """MuJoCo 由 sensorsize + focalpixel 推 fovy；用它反查 fy 是否被采纳。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    cid = _camera_id(model, "hf868_camera")
    fy, height = 326.27539, 480
    expected_fovy = 2.0 * np.degrees(np.arctan(height / (2.0 * fy)))
    assert abs(model.cam_fovy[cid] - expected_fovy) < 0.01


def test_renders_from_calibrated_viewpoint() -> None:
    """从标定位姿必须能渲染出非空画面，且立方体可见。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="hf868_camera")
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    renderer.close()

    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    visible = ((seg[..., 0] == cube_geom) & (seg[..., 1] == 5)).sum()
    assert visible > 200, f"立方体只占 {visible} px，视野可能跑偏了"
