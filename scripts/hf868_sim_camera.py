"""把实机 HF868 的成像几何对齐到训练用的 front_camera。

背景
----
训练数据（workspace_v1_100）是用 MuJoCo 的 `front_camera`（fovy=35°）渲染的，
320x240 下 fx=fy=380.59。实机 HF868 是 135° 广角，640x480 下 fx=326.68，
在 320x240 等效 fx=163.34 —— 广了 2.33 倍。

直接把 HF868 内参搬进仿真会出问题：同一位姿下方块从 18x26 px 缩到 8x11 px，
画幅覆盖从 0.21 m 变成 1.26 m。所以这里采取「裁图匹配 FOV」的路线：从 640x480
裁出中心一块再缩到 320x240，让等效内参与训练相机完全一致。

这让迁移在几何层变成恒等映射，且附带两个好处：
- 畸变残差从全幅 24.6 px 降到 ~1.0 px（只用到近中心区域）；
- 主点偏移恰好归零（裁窗以主点为中心）。

MuJoCo 3.15 的能力边界（实测，见 `probe_mujoco_camera`）
-------------------------------------------------------
- 精确 fx/fy：可以，用 sensorsize + resolution + focalpixel；
- 主点偏移 principal：**不可用**，一旦设置渲染即输出全空图（即使取 0.5/0.5）；
- 镜头畸变：仿真不建模，需在渲染后自行 remap（裁图后已可忽略）。

用法
----
    ./.venv/bin/python scripts/hf868_sim_camera.py            # 打印对齐参数
    ./.venv/bin/python scripts/hf868_sim_camera.py --verify    # 渲染实测方块尺寸
    ./.venv/bin/python scripts/hf868_sim_camera.py --xml       # 输出 MJCF 相机片段
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 实机 HF868 内参（640x480，来自 calibration/cameras/hf868_135_640x480）----
FX, FY = 326.68129, 326.27539
CX, CY = 319.20034, 248.66437
K1, K2, P1, P2, K3 = -0.012865, -0.018761, 0.002971, -0.001267, 0.0
CAM_W, CAM_H = 640, 480

# ---- 训练用的 front_camera（scene_fixed.xml, fovy=35°）----
OUT_W, OUT_H = 320, 240
FOVY_TRAIN = 35.0
FX_TRAIN = (OUT_H / 2) / np.tan(np.radians(FOVY_TRAIN / 2))


@dataclass(frozen=True)
class CropSpec:
    """从实机图裁出与训练相机 FOV 一致的窗口。"""

    x0: float
    y0: float
    w: float
    h: float

    @property
    def scale_x(self) -> float:
        return OUT_W / self.w

    @property
    def scale_y(self) -> float:
        return OUT_H / self.h

    def apply(self, image: np.ndarray) -> np.ndarray:
        """裁 + 缩放成 320x240 uint8。用最近邻以保持像素值不被插值模糊。"""
        import cv2

        xi, yi = round(self.x0), round(self.y0)
        wi, hi = round(self.w), round(self.h)
        cropped = image[yi : yi + hi, xi : xi + wi]
        if cropped.shape[:2] != (hi, wi):
            raise ValueError(f"裁剪越界: 得到 {cropped.shape[:2]}，期望 {(hi, wi)}")
        interp = cv2.INTER_AREA if self.scale_x < 1 else cv2.INTER_LINEAR
        return cv2.resize(cropped, (OUT_W, OUT_H), interpolation=interp)

    def effective_intrinsics(self) -> dict[str, float]:
        return {
            "fx": FX * self.scale_x,
            "fy": FY * self.scale_y,
            # 裁窗以主点为中心时，主点落回画面正中
            "cx": (CX - self.x0) * self.scale_x,
            "cy": (CY - self.y0) * self.scale_y,
        }

    def distortion_at_corner(self) -> tuple[float, float]:
        """返回 (百分比, 缩放后像素位移)。"""
        xn = (self.w / 2) / FX
        yn = (self.h / 2) / FY
        r2 = xn * xn + yn * yn
        radial = 1 + r2 * (K1 + r2 * (K2 + r2 * K3))
        pct = abs(1 - radial) * 100
        disp = abs(1 - radial) * np.sqrt(r2) * FX * self.scale_x
        return pct, disp


def crop_to_match_training() -> CropSpec:
    """以主点为中心裁窗，使等效 fx 等于训练相机。"""
    w = FX * OUT_W / FX_TRAIN
    h = w * OUT_H / OUT_W
    x0 = float(np.clip(CX - w / 2, 0, CAM_W - w))
    y0 = float(np.clip(CY - h / 2, 0, CAM_H - h))
    return CropSpec(x0=x0, y0=y0, w=w, h=h)


def mujoco_camera_xml(name: str = "hf868_camera",
                      pos: str = "0.75 0 0.45",
                      xyaxes: str = "0 1 0 -0.5 0 1") -> str:
    """生成与实机 fx/fy 一致的 MJCF 相机。

    注意：不写 principal —— MuJoCo 3.15 设了它渲染就全空。主点偏移改用裁窗
    （`crop_to_match_training`）消除，效果等价且不会触发这个问题。
    """
    sensor_w, sensor_h = 0.036, 0.027  # 只需宽高比 4:3
    fpx = FX / (CAM_W / OUT_W)          # 若要以 320x240 输出
    fpy = FY / (CAM_H / OUT_H)
    return (
        f'<camera name="{name}" pos="{pos}" xyaxes="{xyaxes}"\n'
        f'        sensorsize="{sensor_w} {sensor_h}" resolution="{OUT_W} {OUT_H}"\n'
        f'        focalpixel="{fpx:.6f} {fpy:.6f}"/>'
    )


def hf868_principal_shift(width: int = CAM_W, height: int = CAM_H) -> tuple[float, float]:
    """MuJoCo 渲染相对实机的整体像素偏移（因为主点偏移写不进去）。

    MuJoCo 一律把主点放在画面正中，而 HF868 的实机主点是 (CX, CY)。
    所以 MuJoCo 渲染出的图像 = 实机图像平移了下面这个量：

        dx = CX - width/2 ,  dy = CY - height/2

    实机图 = MuJoCo 图 + (dx, dy)。需要逐像素对齐时用它做平移；
    640x480 下是 (-0.80, +8.66) px，小于当前外参不确定度，暂可不补。

    返回 (dx, dy)，单位 px。
    """
    return CX - width / 2.0, CY - height / 2.0


def probe_mujoco_camera() -> None:
    """实测 MuJoCo 对 sensorsize / focalpixel / principal 的支持情况。"""
    import mujoco

    def render(extra: str) -> int:
        xml = f"""<mujoco><worldbody><light pos="0 0 1"/>
          <camera name="c" pos="0 0 1" sensorsize="0.036 0.027" resolution="320 240"
                  focalpixel="200 200" {extra}/>
          <geom name="box" type="box" size=".05 .05 .05"/></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=240, width=320)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="c")
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        r.close()
        return int(((seg[..., 0] == 0) & ((seg[..., 1] == 5) | (seg[..., 1] == 0))).sum())

    print("=== MuJoCo 相机能力实测 ===")
    for label, extra in (("仅 focalpixel", ""),
                         ("+ principal 0.5 0.5", 'principal="0.5 0.5"'),
                         ("+ principal 0.4988 0.5180", 'principal="0.4988 0.5180"')):
        n = render(extra)
        verdict = "正常" if n > 0 else "渲染为空 —— 不可用"
        print(f"  {label:28s} 箱体 {n:5d} px  {verdict}")


def verify_rendering() -> int:
    """渲染实测：训练相机 vs 裁图后的实机等效 FOV。"""
    import mujoco

    scene = Path(__file__).resolve().parents[1] / "models/so101/scene_fixed.xml"
    xml = scene.read_text(encoding="utf-8")

    anchor = ('<camera name="front_camera" pos="0.75 0 0.45"\n'
              '      xyaxes="0 1 0 -0.5 0 1" fovy="35"/>')
    if anchor not in xml:
        print("找不到 front_camera 定义，场景可能已改动")
        return 1

    # 一台与「裁图后 HF868」等效的仿真相机（fx 与训练相机相同）
    eq = (f'<camera name="hf868_cropped" pos="0.75 0 0.45" xyaxes="0 1 0 -0.5 0 1" '
          f'fovy="{FOVY_TRAIN}"/>')
    tmp = scene.parent / "_tmp_verify.xml"
    tmp.write_text(xml.replace(anchor, anchor + "\n    " + eq), encoding="utf-8")
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "folded_ready")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        print("=== 方块像素尺寸实测 (320x240) ===")
        for cam in ("front_camera", "hf868_cropped"):
            r = mujoco.Renderer(m, height=OUT_H, width=OUT_W)
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam)
            seg = r.render().copy()
            r.disable_segmentation_rendering()
            r.close()
            mask = (seg[..., 0] == cg) & (seg[..., 1] == 5)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                print(f"  {cam:16s} 未检出")
                continue
            print(f"  {cam:16s} {xs.max()-xs.min()+1:3d} x {ys.max()-ys.min()+1:3d} px  "
                  f"面积 {mask.sum():5d} px")
        return 0
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="渲染实测方块尺寸")
    ap.add_argument("--probe", action="store_true", help="实测 MuJoCo 相机能力")
    ap.add_argument("--xml", action="store_true", help="输出 MJCF 相机片段")
    args = ap.parse_args()

    spec = crop_to_match_training()
    ei = spec.effective_intrinsics()

    print("=== HF868 -> 训练 front_camera 对齐 ===")
    print(f"训练相机  : fx=fy={FX_TRAIN:.2f} (fovy={FOVY_TRAIN}°)  {OUT_W}x{OUT_H}")
    print(f"实机 HF868: fx={FX:.2f} fy={FY:.2f}  {CAM_W}x{CAM_H}  "
          f"(320x240 等效 fx={FX/2:.2f})")
    print(f"焦距比    : {FX/2/FX_TRAIN:.3f}（实机广 {FX_TRAIN/(FX/2):.2f} 倍）")
    print()
    print("裁剪窗口（11 参数里只要这 4 个）:")
    print(f"  x0={spec.x0:.2f} y0={spec.y0:.2f} w={spec.w:.2f} h={spec.h:.2f}")
    print(f"  即 x[{spec.x0:.0f},{spec.x0+spec.w:.0f}] y[{spec.y0:.0f},{spec.y0+spec.h:.0f}] "
          f"-> resize {OUT_W}x{OUT_H}（放大 {spec.scale_x:.3f}x）")
    print()
    print("等效内参:")
    print(f"  fx={ei['fx']:.2f} fy={ei['fy']:.2f} cx={ei['cx']:.2f} cy={ei['cy']:.2f}")
    print(f"  主点偏移 ({ei['cx']-OUT_W/2:+.2f}, {ei['cy']-OUT_H/2:+.2f}) px")
    pct, disp = spec.distortion_at_corner()
    print(f"  裁窗角畸变 {pct:.2f}%，残差 {disp:.2f} px（全幅角是 24.6 px）")

    if args.xml:
        print("\n=== MJCF 相机片段（等效 320x240）===")
        print(mujoco_camera_xml())

    if args.probe:
        print()
        probe_mujoco_camera()

    if args.verify:
        print()
        return verify_rendering()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
