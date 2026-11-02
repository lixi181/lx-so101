"""在 macOS 上使用 mjpython 打开 SO-101 MuJoCo Viewer。"""

import time

import mujoco
from mujoco import viewer

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model


def main() -> None:
    model, data = load_model(FIXED_SCENE_PATH)
    ready_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
    mujoco.mj_resetDataKeyframe(model, data, ready_id)
    mujoco.mj_forward(model, data)
    with viewer.launch_passive(model, data) as window:
        while window.is_running():
            mujoco.mj_step(model, data)
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
