"""验证固定场景下方块能够被物理夹持、抬升并保持。"""

from types import SimpleNamespace

import mujoco.viewer
from module_loader import load_script


def test_fixed_cube_grasp():
    module = load_script("scripts/pick_cube.py", "pick_cube")
    assert module.run(with_viewer=False).success


def test_return_path_does_not_reenter_safe_unfold():
    module = load_script("scripts/pick_cube.py", "pick_cube_return_path")

    for profile in ("simulation_fast", "real_safe"):
        phases = []

        def observer(_model, _data, phase, _action, phases=phases):
            if phase.startswith("回程") or phase == "从盒口撤回":
                phases.append(phase)

        assert module.run(with_viewer=False, observer=observer, profile_name=profile).success
        assert "回程 safe_unfold" not in phases
        assert phases.index("回程 clearance") < phases.index("回程 folded_ready")


class FakeWindow:
    """假的 passive viewer：记录 sync/close，并在等待循环里主动结束。

    `syncs` 就是播放过的 tick 数：演示期间每个 tick 调用一次，演示结束后脚本用
    「sync 直到窗口关闭」的方式停在最终姿态，所以 syncs 会远大于 tick 数。
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.syncs = 0
        self.closed = False
        self.cam = SimpleNamespace(lookat=[0.0, 0.0, 0.0], distance=0.0, azimuth=0.0, elevation=0.0)

    def is_running(self) -> bool:
        return self.syncs < self.limit

    def sync(self) -> None:
        self.syncs += 1

    def close(self) -> None:
        self.closed = True


def test_viewer_path_plays_frames_and_closes_window(monkeypatch):
    """覆盖 with_viewer=True 的分支：设置相机、按 tick 同步、结束时关窗。

    用假窗口替掉 mjpython 的 passive viewer，避免测试里真的弹窗。
    """
    module = load_script("scripts/pick_cube.py", "pick_cube_viewer")
    window = FakeWindow(limit=5000)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args, **_kwargs: window)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module.run(with_viewer=True)

    assert result.success
    assert window.cam.distance > 0.0, "viewer 初始化时应该设置相机"
    assert window.syncs > 1000, "演示期间要按 tick 同步，结束后停在最终姿态"
    assert window.closed, "run() 退出时必须关闭窗口"
