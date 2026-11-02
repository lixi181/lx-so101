import mujoco

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH


def test_folded_ready_is_fixed_start_pose() -> None:
    model = mujoco.MjModel.from_xml_path(str(FIXED_SCENE_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
    assert key_id >= 0

    mujoco.mj_resetDataKeyframe(model, data, key_id)

    expected = [-0.0187914029, -1.68, 1.56, 1.18, 0.0, 0.0]
    assert list(data.qpos[:6]) == expected
    assert list(data.ctrl[:6]) == expected
