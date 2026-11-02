"""相机内参与投影/反投影。

坐标约定统一为 OpenCV 相机系：x 向右、y 向下、z 沿光轴向前（看向 +z）。
MuJoCo 的相机系是 x 右、y 上、z 朝后，两者只差 y 和 z 的符号，
需要跨库换算时用 `cam_to_mujoco` / `mujoco_to_cam`。

畸变模型是 ROS 的 plumb_bob（k1 k2 p1 p2 k3）。这里用 numpy 实现，
不依赖 OpenCV —— pyproject 里没有 opencv，标定脚本那边才有。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "references" / "project_metadata" / "configs" / "camera"

# 真实机械臂的标定数据放在 so_arm_ws，不在本仿真工作区。so_arm_ws 借用本工作区的
# Python 环境（../so_ws/.venv/bin/python），所以运行时的 cwd 通常就是 so_arm_ws。
# 搜索顺序：显式路径 > $SO_ARM_WS > 常见 so_arm_ws 位置 > 本仓库副本。
# 仓库副本是版本控制的镜像，供没有 so_arm_ws 的开发机（如本机 macOS）使用。
SO_ARM_WS_ENV = "SO_ARM_WS"


def candidate_camera_dirs() -> list[Path]:
    """按优先级列出可能存放内参的目录。"""
    dirs: list[Path] = []

    env = os.environ.get(SO_ARM_WS_ENV)
    if env:
        dirs.append(Path(env) / "calibration" / "cameras")

    # 运行时 cwd 就在 so_arm_ws 里的情况（README 推荐从那里启动）
    dirs.append(Path.cwd() / "calibration" / "cameras")

    home = Path.home()
    dirs.append(home / "so_projects" / "so_arm_ws" / "calibration" / "cameras")
    dirs.append(home / "so_arm_ws" / "calibration" / "cameras")

    # 本仓库内的版本控制镜像，放最后
    dirs.append(CONFIG_DIR)

    seen: set[Path] = set()
    unique = []
    for d in dirs:
        resolved = d.resolve() if d.exists() else d
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(d)
    return unique



@dataclass(frozen=True)
class CameraModel:
    """单目针孔相机（含径向+切向畸变）。"""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / 2.0 / self.fx)))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / 2.0 / self.fy)))

    # ---------------------------------------------------------------- 畸变

    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        """理想归一化坐标 -> 含畸变的归一化坐标。输入 (..., 2)。"""
        xy = np.asarray(xy, dtype=float)
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return np.stack([x * radial + dx, y * radial + dy], axis=-1)

    def undistort_normalized(self, xy_dist: np.ndarray, iterations: int = 20) -> np.ndarray:
        """含畸变归一化坐标 -> 理想归一化坐标（不动点迭代）。

        常规镜下收敛很快；20 次迭代后残差远小于 0.01px。
        """
        xy_dist = np.asarray(xy_dist, dtype=float)
        xy = xy_dist.copy()
        for _ in range(iterations):
            xy = xy_dist - (self.distort_normalized(xy) - xy)
        return xy

    # ---------------------------------------------------------------- 投影

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """相机系 3D 点 (..., 3) -> 像素坐标 (..., 2)（含畸变，符合实际成像）。"""
        p = np.asarray(points_cam, dtype=float)
        if np.any(p[..., 2] <= 0):
            raise ValueError("点必须在相机前方（z > 0）")
        xy = p[..., :2] / p[..., 2:3]
        xd = self.distort_normalized(xy)
        return np.stack([self.fx * xd[..., 0] + self.cx,
                         self.fy * xd[..., 1] + self.cy], axis=-1)

    def project_ideal(self, points_cam: np.ndarray) -> np.ndarray:
        """不带畸变的投影，用于和已去畸变的图像对照。"""
        p = np.asarray(points_cam, dtype=float)
        if np.any(p[..., 2] <= 0):
            raise ValueError("点必须在相机前方（z > 0）")
        xy = p[..., :2] / p[..., 2:3]
        return np.stack([self.fx * xy[..., 0] + self.cx,
                         self.fy * xy[..., 1] + self.cy], axis=-1)

    def unproject(self, pixels: np.ndarray, depth_m: float | np.ndarray) -> np.ndarray:
        """像素坐标 (..., 2) + 深度（沿光轴 z）-> 相机系 3D 点 (..., 3)。"""
        px = np.asarray(pixels, dtype=float)
        xd = np.stack([(px[..., 0] - self.cx) / self.fx,
                       (px[..., 1] - self.cy) / self.fy], axis=-1)
        xy = self.undistort_normalized(xd)
        z = np.broadcast_to(np.asarray(depth_m, dtype=float), xy.shape[:-1])
        return np.stack([xy[..., 0] * z, xy[..., 1] * z, z], axis=-1)

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """把实拍像素坐标映射到理想（无畸变）像素坐标。"""
        px = np.asarray(pixels, dtype=float)
        xd = np.stack([(px[..., 0] - self.cx) / self.fx,
                       (px[..., 1] - self.cy) / self.fy], axis=-1)
        xy = self.undistort_normalized(xd)
        return np.stack([self.fx * xy[..., 0] + self.cx,
                         self.fy * xy[..., 1] + self.cy], axis=-1)

    def pixel_ray(self, pixel: np.ndarray) -> np.ndarray:
        """像素对应的单位方向向量（相机系），常用于点云/射线求交。"""
        p = self.unproject(np.asarray(pixel, dtype=float), 1.0)
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    # ---------------------------------------------------------------- 坐标换算

    @staticmethod
    def cam_to_mujoco(points_cam: np.ndarray) -> np.ndarray:
        """OpenCV 相机系 -> MuJoCo 相机系（y 取反、z 取反）。"""
        p = np.array(points_cam, dtype=float, copy=True)
        p[..., 1] *= -1.0
        p[..., 2] *= -1.0
        return p

    @staticmethod
    def mujoco_to_cam(points_mj: np.ndarray) -> np.ndarray:
        """MuJoCo 相机系 -> OpenCV 相机系。变换是对合的，与上者同一操作。"""
        return CameraModel.cam_to_mujoco(points_mj)


def resolve_camera_yaml(camera_dir: str | Path) -> Path:
    """在候选目录中定位 <camera_dir>/intrinsics.yaml，找不到就报清楚。"""
    name = str(camera_dir)
    tried = []
    for base in candidate_camera_dirs():
        path = base / name / "intrinsics.yaml"
        tried.append(path)
        if path.is_file():
            return path
    listing = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        f"找不到相机内参 {name}/intrinsics.yaml。已尝试：\n{listing}\n"
        f"真实机械臂的标定应放在 <so_arm_ws>/calibration/cameras/{name}/，"
        f"或用环境变量 {SO_ARM_WS_ENV} 指定 so_arm_ws 根目录。"
    )


def load_camera(camera_dir: str | Path = "hf868_135_640x480") -> CameraModel:
    """按目录名加载内参，自动在 so_arm_ws 与本仓库副本之间查找。"""
    return load_camera_from_yaml(resolve_camera_yaml(camera_dir))


def load_camera_from_yaml(path: str | Path) -> CameraModel:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    intr = data["intrinsics"]
    dist = data["distortion"]
    res = data["resolution"]
    return CameraModel(
        name=data.get("model", {}).get("name", Path(path).parent.name),
        width=int(res["width"]),
        height=int(res["height"]),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        k1=float(dist.get("k1", 0.0)),
        k2=float(dist.get("k2", 0.0)),
        p1=float(dist.get("p1", 0.0)),
        p2=float(dist.get("p2", 0.0)),
        k3=float(dist.get("k3", 0.0)),
    )


def load_from_ros_ost(path: str | Path) -> CameraModel:
    """读取 ROS camera_calibration 的 ost.yaml（usb_cam 用的那种格式）。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=float).ravel()
    if data.get("distortion_model", "plumb_bob") != "plumb_bob":
        raise ValueError(f"只支持 plumb_bob，收到 {data['distortion_model']}")
    k1, k2, p1, p2, k3 = (list(D) + [0.0] * 5)[:5]
    return CameraModel(
        name=str(data.get("camera_name", Path(path).parent.name)),
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
    )
