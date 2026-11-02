from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).resolve().parents[4] / "models" / "so101" / "scene.xml"
FIXED_SCENE_PATH = Path(__file__).resolve().parents[4] / "models" / "so101" / "scene_fixed.xml"


def load_model(path: Path = MODEL_PATH) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载工作空间内的 MuJoCo 场景并创建仿真状态。"""
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def scene_timestep(path: Path = FIXED_SCENE_PATH) -> float:
    """读取场景的物理步长。

    控制周期由 `physics_per_tick × timestep` 算出，所以这个值必须从场景里读，
    不能在任何模块里写死一个步长常量；否则改了 XML 之后模块级默认值会悄悄失配。
    """
    return float(mujoco.MjModel.from_xml_path(str(path)).opt.timestep)
