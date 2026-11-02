"""检查 front / wrist 相机在采样范围四个角和中心的取景，并测量方块在 front 图中的像素尺寸。

方块像素尺寸的测法：把方块 geom 的 alpha 临时置 0 再渲染同一帧，两次渲染的差异就是方块
的可见轮廓。这样既不需要颜色阈值（夹爪的红色和方块很接近），也不依赖相机内参约定 ——
测出来的就是策略实际拿到的那张图里方块的像素数。

测量时必须临时关掉 `mjRND_REFLECTION`：桌面材质有 `reflectance=0.15`，倒影和方块一起
参与像素差异，会把轮廓高度从 19 px 撑到 29 px（实测），面积虚高 56%。关掉阴影没有影响，
因为桌面在这里不产生可见硬阴影。

用法：

    ./.venv/bin/python scripts/check_camera_framing.py
    ./.venv/bin/python scripts/check_camera_framing.py --output-dir results/camera_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pick_cube
from PIL import Image

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model
from task_sampling import SCENE_BOX_XY, TaskSample

# 采样范围 r ∈ (0.20, 0.24) m、θ ∈ (-30°, 30°) 的四个角，外加中心一点。
CORNER_RADII_M = (0.20, 0.24)
CORNER_AZIMUTHS_DEG = (-30.0, 30.0)
CENTER_RADIUS_M = 0.22
CUBE_EDGE_M = 0.03
CAMERAS = ("front_camera", "wrist_camera")
CUBE_GEOM = "cube_geom"
# 方块投影后的单边长度低于这个值就说明 front 取景不够紧，需要调整相机位置或 fovy。
MIN_CUBE_EDGE_PX = 12.0


def framing_tasks() -> list[tuple[str, TaskSample]]:
    """四个角和中心各一个任务；角点用采样范围边界，中心取半径中点、方位角 0。"""
    positions: list[tuple[str, float, float]] = [
        (f"corner_r{radius:.2f}_a{azimuth:+.0f}", radius, azimuth)
        for radius in CORNER_RADII_M
        for azimuth in CORNER_AZIMUTHS_DEG
    ]
    positions.append(("center", CENTER_RADIUS_M, 0.0))
    tasks = []
    for name, radius, azimuth_deg in positions:
        azimuth = float(np.deg2rad(azimuth_deg))
        tasks.append(
            (
                name,
                TaskSample(
                    cube_xy=(radius * float(np.cos(azimuth)), radius * float(np.sin(azimuth))),
                    cube_yaw=0.0,
                    box_xy=SCENE_BOX_XY,
                    seed=0,
                ),
            )
        )
    return tasks


def cube_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: str,
    cube_geom_id: int,
) -> np.ndarray:
    """返回方块在该相机下的可见轮廓。

    渲染两次：正常一次，把方块 geom 的 alpha 置 0 再一次。`model.geom_rgba` 是渲染直接
    读取的，改完立刻生效，不需要重新 mj_forward。第二次渲染前关掉桌面反射，否则倒影会
    被算进轮廓。
    """
    flags = renderer.scene.flags
    original_reflection = int(flags[mujoco.mjtRndFlag.mjRND_REFLECTION])
    original_alpha = float(model.geom_rgba[cube_geom_id, 3])
    try:
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        renderer.update_scene(data, camera=camera)
        with_cube = renderer.render().copy()
        model.geom_rgba[cube_geom_id, 3] = 0.0
        renderer.update_scene(data, camera=camera)
        without_cube = renderer.render().copy()
    finally:
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = original_reflection
        model.geom_rgba[cube_geom_id, 3] = original_alpha
    return np.any(with_cube != without_cube, axis=2)


def silhouette_metrics(mask: np.ndarray) -> dict:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return {"visible": False}
    return {
        "visible": True,
        "width_px": int(cols.max() - cols.min() + 1),
        "height_px": int(rows.max() - rows.min() + 1),
        "area_px": int(mask.sum()),
        # 面积等效边长：把轮廓当成正方形，给一个可比较的单值。
        "equivalent_edge_px": float(np.sqrt(mask.sum())),
    }


def camera_depth_m(model: mujoco.MjModel, data: mujoco.MjData, camera: str, point: np.ndarray) -> float:
    """点到相机光轴的距离（MuJoCo 相机看向自身局部 -z）。"""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"找不到相机: {camera}")
    rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    return float(-(rotation.T @ (point - data.cam_xpos[camera_id]))[2])


def analytic_edge_px(fovy_deg: float, height: int, depth_m: float, edge_m: float) -> float:
    """针孔模型的单边参考值：f = (H/2)/tan(fovy/2)，边长 ≈ f × 边长 / 距离。

    MuJoCo 的 `cam_fovy` 是竖直视场角：实测把方块沿世界 y 平移 20 mm，质心像素位移
    反推 f ≈ 297.7 px，与 (240/2)/tan(22.5°) = 289.7 px 一致（差 2.8%，来自透视和
    质心估计误差），而水平约定会给出 386.3 px，明显不符。
    """
    focal = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return float(focal * edge_m / depth_m)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/camera_check"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, data = load_model(FIXED_SCENE_PATH)
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pick_cube.CUBE_BODY)
    if cube_geom_id < 0 or cube_body_id < 0:
        raise ValueError(f"场景里找不到 {CUBE_GEOM} / {pick_cube.CUBE_BODY}")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    fovy = float(model.cam_fovy[front_camera_id])

    print(
        f"分辨率 {args.width}×{args.height}，front_camera fovy={fovy:g}°，"
        f"方块边长 {CUBE_EDGE_M * 1000:.0f} mm"
    )
    print()
    header = (
        f"{'位置':<20}{'r(m)':>6}{'θ(°)':>7}{'front 距离':>11}"
        f"{'轮廓宽×高':>12}{'面积':>7}{'等效边长':>10}{'针孔单边':>10}{'wrist':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, task in framing_tasks():
        pick_cube.reset_to_ready(model, data, task)
        cube_center = np.asarray(data.xpos[cube_body_id]).copy()

        # 保存给人工看的图用正常设置（带倒影），测量用关掉倒影的掩码。
        for camera in CAMERAS:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render().copy()).save(args.output_dir / f"{name}_{camera}.png")

        front = silhouette_metrics(cube_mask(model, data, renderer, "front_camera", cube_geom_id))
        wrist = silhouette_metrics(cube_mask(model, data, renderer, "wrist_camera", cube_geom_id))
        depth = camera_depth_m(model, data, "front_camera", cube_center)
        reference = analytic_edge_px(fovy, args.height, depth, CUBE_EDGE_M)
        print(
            f"{name:<20}{task.radius_m:>6.2f}{task.azimuth_deg:>7.1f}{depth:>11.3f}"
            f"{front['width_px']:>7}×{front['height_px']:<5}{front['area_px']:>7}"
            f"{front['equivalent_edge_px']:>10.1f}{reference:>10.1f}"
            f"{'是' if wrist['visible'] else '否':>7}"
        )
        rows.append((name, task, depth, front, wrist, reference))

    renderer.close()

    print()
    print(f"图片写到 {args.output_dir}/（每个位置 front + wrist 各一张，正常设置含桌面倒影）")
    print("注意：轮廓高比宽大是因为俯视角度下能看到方块顶面；面积等效边长偏大是因为轮廓不是正方形。")
    # 判定取最保守的两个读数：轮廓宽度（方块的横向单边）和针孔单边参考。
    worst = min(rows, key=lambda row: min(row[3]["width_px"], row[5]))
    conservative_px = min(worst[3]["width_px"], worst[5])
    print(
        f"\n最保守读数在 {worst[0]}（距离 {worst[2]:.3f} m）："
        f"轮廓宽 {worst[3]['width_px']} px，针孔单边参考 {worst[5]:.1f} px"
    )
    print(f"阈值 {MIN_CUBE_EDGE_PX:.0f} px → {'达标' if conservative_px >= MIN_CUBE_EDGE_PX else '不达标'}")
    if conservative_px < MIN_CUBE_EDGE_PX:
        print("方块投影边长低于阈值，需要提出 front_camera 位置或 fovy 的调整方案等确认。")
        return 1
    print(
        f"最保守读数比阈值高 {conservative_px - MIN_CUBE_EDGE_PX:.1f} px"
        f"（相对余量 {(conservative_px / MIN_CUBE_EDGE_PX - 1) * 100:.0f}%）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
