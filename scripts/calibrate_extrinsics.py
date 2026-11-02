"""HF868 相机外参标定：解出相机在机械臂基座坐标系里的位姿。

要解决什么
----------
内参（焦距/主点/畸变）只决定「像素怎么映射到方向」，不决定「相机在哪、朝哪」。
要把实机相机放进仿真（`scene_fixed.xml` 的 `<camera>`），必须知道它在**基座坐标系**
里的位姿。

基座坐标系（与 `scene_fixed.xml` 完全一致，已用模型核对）
--------------------------------------------------------
- 原点：`shoulder_pan` 旋转轴与桌面的交点（基座转台轴心在地面的投影）
- +x：`qpos` 全零时手臂指向的水平方向（仿真里此时夹爪在 (0.293, 0.000, 0.234)，
  即严格沿 +x）
- +z：竖直向上（桌面顶面 z=0）
- +y：右手系，即面朝 +x 时指向左侧（仿真里 target_box 在 y=+0.18）

标定原理
--------
1. 棋盘格平放桌面 → 板法向就是 +z，板→基座只剩 3 个未知量 `(x0, y0, yaw)`；
2. 手工量出板上三个角点相对基座轴心的位置，直接定出板的位姿（精确，非拟合）；
3. 于是所有内角点在基座系里的 3D 坐标都是已知的，`solvePnP` 一次得到相机位姿。

⚠ 为什么需要「臂在哪一侧」
--------------------------
平放在桌面上的棋盘格是**对称靶标**：把它绕自身法向转 180°，点集与自身重合。
因此单张图 + 单个平面靶标存在固有的 180° 二义性 —— 两个解的重投影残差都是 0，
生成的图像完全一样（实测两个候选残差都是 0.000 px）。镜像解会落到桌面以下，
可以自动排除；剩下 180° 那个必须靠**平面外的东西**来区分。

机械臂基座正好在棋盘平面外，所以只要回答一句「照片里机械臂在画面哪一侧」，
工具就能自动选出唯一解（`--arm-side`）。

用法
----
    # 0) 自检：合成数据验证整条数学链路，不需要相机
    ./.venv/bin/python scripts/calibrate_extrinsics.py --selftest

    # 1) 用一张已拍好的照片
    ./.venv/bin/python scripts/calibrate_extrinsics.py \\
        --image photo.png --square-mm 25 \\
        --points "180,120 420,120 180,400" \\
        --arm-side left

    # 2) 直接从相机抓图（偏暗时自动长曝光 + 多帧择优）
    ./.venv/bin/python scripts/calibrate_extrinsics.py \\
        --capture /dev/video0 --square-mm 25 \\
        --points "180,120 420,120 180,400" --arm-side left

`--points` 的三点顺序不能乱，定义见 `--help` 与下方 `board_pose_from_points`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# 允许用环境变量指定输出位置：脚本被拷到仓库外运行时 parents[1] 会退化成 "/"，
# 此时默认路径不可写。
RESULT_PATH = Path(os.environ.get("EXTRINSICS_OUT",
                                  REPO_ROOT / "results" / "extrinsics_hf868.json"))

# 实机 HF868 内参（640x480，来自 calibration/cameras/hf868_135_640x480）
FX, FY = 326.68129, 326.27539
CX, CY = 319.20034, 248.66437
DIST = np.array([-0.012865, -0.018761, 0.002971, -0.001267, 0.0])
CAM_W, CAM_H = 640, 480

# 棋盘格：8x11 方格 → 7x10 内角点
COLS, ROWS = 7, 10


# --------------------------------------------------------------------- 基础


@dataclass(frozen=True)
class BoardPose:
    """棋盘格在基座坐标系里的位姿（平放桌面，法向 +z）。"""

    x0: float
    y0: float
    z0: float          # 内角点所在平面高度（板厚）
    yaw: float         # 绕 +z 的偏航角，弧度

    @property
    def R(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def t(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.z0])


def board_object_points(square_m: float, cols: int = COLS, rows: int = ROWS) -> np.ndarray:
    """棋盘格内角点在板坐标系里的 3D 坐标（米，z=0），按行优先排列。"""
    pts = np.zeros((rows * cols, 3), np.float64)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def normalize_handedness(measured_m: np.ndarray, cols: int, rows: int
                         ) -> tuple[np.ndarray, int, int, dict]:
    """保证「+X × +Y = +z」。板子平放桌面时这是几何必然。

    给定三点 P0/P1/P2，若 P0→P1 与 P0→P2 在桌面平面内构成**左手系**，说明轴
    归属反了：P1 其实沿短边、P2 沿长边。此时交换 P1/P2 并交换 cols/rows。

    这不是「容错」而是纠错 —— 左手系的板子在右手系里不存在，出现即表示
    标注顺序与实物轴不一致。自动纠正后仍会打印说明，便于核对。
    """
    info = {"swapped": False}
    if len(measured_m) < 3:
        return measured_m, cols, rows, info
    ex = measured_m[1, :2] - measured_m[0, :2]
    ey = measured_m[2, :2] - measured_m[0, :2]
    lx, ly = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    if lx < 1e-9 or ly < 1e-9:
        return measured_m, cols, rows, info
    cross_z = (ex[0] * ey[1] - ex[1] * ey[0]) / (lx * ly)
    info["cross_z"] = round(float(cross_z), 4)
    if cross_z < 0:
        out = measured_m.copy()
        out[1], out[2] = measured_m[2].copy(), measured_m[1].copy()
        info["swapped"] = True
        return out, rows, cols, info
    return measured_m, cols, rows, info


def board_pose_from_points(measured_m: np.ndarray, square_m: float,
                           cols: int = COLS, rows: int = ROWS,
                           z0: float = 0.0) -> tuple[BoardPose, dict]:
    """由 2 个（或 3 个）标记内角点定出棋盘格在基座坐标系里的位姿。

    板平放桌面，法向 +z，所以只剩 `(x0, y0, yaw)` 三个未知量：
      - P0 的 (x, y) 直接给出 x0, y0；
      - P0→P1 的方向给出 yaw；
      - z0 由板厚给出。
    因此 **2 个点就足够**，第 3 个点只用来交叉校验（垂直度）。

    P0、P1 必须是内角点网格的**角上**两点（不能是内部点），且沿板的某一条边。
    P0→P1 走的是长边还是短边不需要你说明 —— 长边 (cols-1)·s 与短边 (rows-1)·s
    相差很大（例如 225 vs 150 mm），实测距离能自动分辨。
    """
    p0, p1 = measured_m[0, :2], measured_m[1, :2]
    ex = p1 - p0
    lx = float(np.linalg.norm(ex))
    want_x, want_y = (cols - 1) * square_m, (rows - 1) * square_m

    # 判断 P0→P1 沿的是哪条轴：比较实测长度与两个预期长度的差距
    along_x = abs(lx - want_x) <= abs(lx - want_y)

    checks = {
        "n_points": len(measured_m),
        "p0_p1_mm": round(lx * 1000, 2),
        "expected_along_x_mm": round(want_x * 1000, 2),
        "expected_along_y_mm": round(want_y * 1000, 2),
        "p0_p1_along": "cols轴(+X)" if along_x else "rows轴(+Y)",
        "p0_p1_err_mm": round((lx - (want_x if along_x else want_y)) * 1000, 2),
    }

    if along_x:
        yaw = float(np.arctan2(ex[1], ex[0]))
    else:
        # P0→P1 沿板的 +Y。板 +Y 在基座系里的单位向量是 u，则 +X = (u_y, -u_x)。
        ux, uy = ex[0] / lx, ex[1] / lx
        yaw = float(np.arctan2(-ux, uy))

    board = BoardPose(x0=float(p0[0]), y0=float(p0[1]), z0=float(z0), yaw=yaw)

    # 可选的第 3 点：检查它是否落在预期位置（垂直边端点）
    if len(measured_m) >= 3:
        p2 = measured_m[2, :2]
        ey = p2 - p0
        ly = float(np.linalg.norm(ey))
        want_other = want_y if along_x else want_x
        checks["p0_p2_mm"] = round(ly * 1000, 2)
        checks["p0_p2_expected_mm"] = round(want_other * 1000, 2)
        checks["p0_p2_err_mm"] = round((ly - want_other) * 1000, 2)
        if lx > 1e-9 and ly > 1e-9:
            checks["orthogonality_err_deg"] = round(float(
                np.degrees(np.arccos(np.clip(ex @ ey / (lx * ly), -1.0, 1.0))) - 90.0), 2)
    return board, checks


def layout_warnings(checks: dict) -> list[str]:
    """把量测误差翻译成可执行的提醒。"""
    out = []
    if abs(checks["p0_p1_err_mm"]) > 8:
        along = checks["p0_p1_along"]
        exp = (checks["expected_along_x_mm"] if "cols" in along
               else checks["expected_along_y_mm"])
        out.append(f"P0→P1 实测 {checks['p0_p1_mm']:.0f} mm，"
                   f"按{along}应为 {exp:.0f} mm（差 {checks['p0_p1_err_mm']:+.0f} mm）"
                   " —— 可能量错了点，或方格边长不是标称值")
    if "orthogonality_err_deg" in checks and abs(checks["orthogonality_err_deg"]) > 3:
        out.append(f"P0→P1 与 P0→P2 夹角偏差 {checks['orthogonality_err_deg']:+.1f}°，"
                   "应接近 0（这两条边在板上互相垂直）")
    return out


# --------------------------------------------------------------------- 检测


def detect_corners(image: np.ndarray, cols: int = COLS, rows: int = ROWS,
                   clahe: bool = True) -> np.ndarray | None:
    """检测内角点。房间偏暗，所以默认先做 CLAHE 提局部对比度。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    # 现场是弱光：实测在原始分辨率 + CLAHE 3.0 下完全检不出，而「CLAHE 4.0 + 2 倍
    # 上采样」可以稳定检出 70 点，所以这里按「先强后弱」的顺序逐档尝试。
    # 同时尝试 (cols,rows) 与 (rows,cols)：板子被转 90° 摆放时只有后者能匹配。
    attempts = []
    for size in ((cols, rows), (rows, cols)):
        for clip, scale in ((4.0, 2.0), (6.0, 2.0), (4.0, 3.0), (3.0, 1.0)):
            attempts.append((size, clip, scale))

    seen = set()
    for size, clip, scale in attempts:
        key = (size, clip, scale)
        if key in seen:
            continue
        seen.add(key)
        base = gray if clip <= 0 else cv2.createCLAHE(
            clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
        im = base if scale == 1.0 else cv2.resize(
            base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, corners = cv2.findChessboardCorners(im, size, flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(im, corners, (11, 11), (-1, -1), crit)
        out = corners.reshape(-1, 2) / scale
        detect_corners.last = {"size": size, "clip": clip, "scale": scale}
        return out
    return None


# --------------------------------------------------------------------- 求解


def lattice_symmetries(cols: int, rows: int) -> list[tuple[str, np.ndarray]]:
    """枚举把内角点格点**映射到自身**的索引置换。

    矩形格点（cols≠rows）只有 4 个：恒等、180°、沿 X 镜像、沿 Y 镜像。
    90° 旋转会把长宽互换，不是合法对称。每个置换给出 obj[perm[k]] 与第 k 个
    检测角点配对。
    """
    a = np.arange(cols)
    b = np.arange(rows)
    aa, bb = np.meshgrid(a, b, indexing="xy")     # aa 快变 -> 行优先

    def build(ma, mb):
        return (ma + mb * cols).reshape(-1)

    return [
        ("恒等", build(aa, bb)),
        ("180°", build(cols - 1 - aa, rows - 1 - bb)),
        ("沿X镜像", build(cols - 1 - aa, bb)),
        ("沿Y镜像", build(aa, rows - 1 - bb)),
    ]


def solve_camera_pose(X_base: np.ndarray, img_pts: np.ndarray,
                      K: np.ndarray, dist: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """solvePnP，返回相机→基座的 (R, t)。

    OpenCV 的 PnP 给的是 X_cam = R_bc @ X_base + t_bc（基座→相机）。
    这里转成相机→基座，便于直接写 MJCF 的 pos/xyaxes。
    """
    ok, rvec, tvec = cv2.solvePnP(X_base, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R_bc, _ = cv2.Rodrigues(rvec)
    t_bc = tvec.reshape(3)
    return R_bc.T, -R_bc.T @ t_bc        # (相机→基座, 相机位置)


def plausible(R_cb: np.ndarray, t_cb: np.ndarray) -> tuple[bool, str]:
    """物理合理性：相机应在桌面上方、镜头朝下、离基座不远。"""
    if not (0.15 <= t_cb[2] <= 1.6):
        return False, f"高度 {t_cb[2]:+.2f} m（应在桌面之上）"
    fwd = R_cb[:, 2]                 # OpenCV：+z 是光轴朝前
    if fwd[2] > -0.10:
        return False, f"光轴竖直分量 {fwd[2]:+.2f}，没朝下"
    if np.linalg.norm(t_cb[:2]) > 1.5:
        return False, f"离基座 {np.linalg.norm(t_cb[:2]):.2f} m 太远"
    return True, ""


def reprojection_report(R_cb: np.ndarray, t_cb: np.ndarray, K: np.ndarray,
                        dist: np.ndarray, X_base: np.ndarray,
                        img_pts: np.ndarray) -> dict:
    """把基座系 3D 点投影回图像，统计与检测值的偏差。"""
    rvec, _ = cv2.Rodrigues(R_cb.T)          # 基座→相机
    tvec = (-R_cb.T @ t_cb).reshape(3, 1)
    proj, _ = cv2.projectPoints(X_base, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
    return {
        "mean_px": float(err.mean()),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(err.max()),
    }


def build_candidates(board: BoardPose, square_m: float, img_pts: np.ndarray,
                     K: np.ndarray, dist: np.ndarray,
                     cols: int = COLS, rows: int = ROWS) -> list[dict]:
    """在 4 种格点对称下各解一次，返回全部候选及其评分。"""
    obj = board_object_points(square_m, cols, rows)
    X_base = obj @ board.R.T + board.t
    cands = []
    for name, perm in lattice_symmetries(cols, rows):
        sol = solve_camera_pose(X_base[perm], img_pts, K, dist)
        if sol is None:
            continue
        R_cb, t_cb = sol
        ok, why = plausible(R_cb, t_cb)
        rep = reprojection_report(R_cb, t_cb, K, dist, X_base[perm], img_pts)
        # 基座原点投影到画面哪里：用来回答「臂在哪一侧」
        rvec, _ = cv2.Rodrigues(R_cb.T)
        tvec = (-R_cb.T @ t_cb).reshape(3, 1)
        base_px, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist)
        cands.append({
            "name": name, "R_cb": R_cb, "t_cb": t_cb,
            "plausible": ok, "reason": why,
            "resid_px": rep["mean_px"], "report": rep,
            "base_px": base_px.reshape(2),
        })
    return cands


def choose_candidate(cands: list[dict], arm_side: str | None,
                     img_w: int) -> tuple[dict, list[str]]:
    """单个视角下选解。

    单视角对平面靶标有固有二义性，`--arm-side` 只能作为兜底判据，而且当基座
    投影靠近画面中心时它并不可靠（`front_camera` 的光轴几乎正对基座，实测
    基座投影恒在 x≈319px）。**优先用 `--image2/--points2` 的双视角一致性。**
    """
    notes = []
    valid = [c for c in cands if c["plausible"]]
    notes.append(f"4 种格点对称下：{len(valid)} 个物理合理")
    if not valid:
        raise RuntimeError(
            "没有任何候选给出物理合理的相机位姿。请检查：\n"
            "  - 棋盘格是否平放桌面（法向朝上）\n"
            "  - --points 的三点顺序是否按约定（P0 原点、P1 沿X、P2 沿Y）\n"
            "  - 坐标是否以基座轴心为原点、单位是否 mm\n"
            "  - 方格边长是否量对"
        )
    if len(valid) == 1:
        notes.append(f"唯一解：{valid[0]['name']}（其余候选落在桌面以下或朝上）")
        return valid[0], notes

    if arm_side is None:
        desc = "\n".join(
            f"    {c['name']:8s} 残差 {c['resid_px']:.3f}px  "
            f"相机 ({c['t_cb'][0]:+.3f}, {c['t_cb'][1]:+.3f}, {c['t_cb'][2]:+.3f})"
            for c in valid)
        raise RuntimeError(
            f"有 {len(valid)} 个候选都物理合理（残差都是 "
            f"{valid[0]['resid_px']:.3f}px）：\n{desc}\n\n"
            "  棋盘格是对称靶标，绕法向转 180° 后点集与自身重合，所以单张图存在\n"
            "  固有的 180° 二义性，且两个解生成的图像完全一样。\n\n"
            "  推荐解法：换一个位置再拍一张，加 --image2 与 --points2。\n"
            "  板移动后正确解的相机位姿不变，错误解会跟着板跑，能自动判定。\n\n"
            "  兜底：--arm-side left|right（但基座常落在画面中心附近，不太可靠）"
        )

    want_left = arm_side == "left"
    seps = [abs(c["base_px"][0] - img_w / 2) for c in valid]
    if min(seps) < 60:
        notes.append(
            f"！注意：候选的基座投影都靠近画面中心（偏离 "
            f"{min(seps):.0f}px），--arm-side 判据很不可靠，"
            "建议改用 --image2/--points2 双视角")
    for c in valid:
        if (c["base_px"][0] < img_w / 2) == want_left:
            notes.append(f"按 --arm-side {arm_side} 选定：{c['name']}"
                         f"（基座投影 x={c['base_px'][0]:.0f}px）")
            return c, notes
    notes.append(f"没有候选符合 --arm-side {arm_side}，退回残差最小的")
    return min(valid, key=lambda c: c["resid_px"]), notes


def _pose_dist(c1: dict, c2: dict) -> float:
    """两个相机位姿的位置差（毫米）。"""
    return float(np.linalg.norm(c1["t_cb"] - c2["t_cb"]) * 1000)


def resolve_two_views(view_cands: list[list[dict]]) -> tuple[dict, list[str]]:
    """用两个视角的位姿一致性破平面靶标的二义性。

    不按对称名配对 —— `findChessboardCorners` 的角点起始顺序会随板的朝向变化，
    同一物理对称在两视角下可能叫不同的名字。所以直接对所有候选两两配对，
    取「位姿间距最小」的那一对。正确解释下两个视角解出的相机位姿几乎相同，
    错误解释会随板移动而分散。
    """
    if len(view_cands) < 2:
        raise ValueError("至少需要两个视角")
    notes = []
    for i, cands in enumerate(view_cands):
        valid = [c for c in cands if c["plausible"]]
        notes.append(f"视角{i + 1}：{len(valid)}/{len(cands)} 个候选物理合理")
        if not valid:
            raise RuntimeError(
                f"视角{i + 1} 没有物理合理的候选。检查棋盘格是否平放桌面、"
                "三点顺序是否符合约定、坐标是否以基座轴心为原点（mm）")

    pairs = []
    for c1 in [c for c in view_cands[0] if c["plausible"]]:
        for c2 in [c for c in view_cands[1] if c["plausible"]]:
            pairs.append((_pose_dist(c1, c2), c1, c2))
    pairs.sort(key=lambda x: x[0])

    best_d, best1, best2 = pairs[0]
    notes.append("位姿一致性配对（间距越小越可信）：")
    for d, a, b in pairs:
        notes.append(f"  {a['name']:8s} + {b['name']:8s} -> {d:8.2f} mm")
    if len(pairs) > 1 and pairs[1][0] < 3.0 * max(best_d, 1.0):
        notes.append(
            f"！注意：最优与次优间距接近（{best_d:.2f} vs {pairs[1][0]:.2f} mm），"
            "判据不明确。建议把板挪得更远重拍第二张（位移 > 100 mm）")
    notes.append(f"选定：{best1['name']} + {best2['name']}，"
                 f"两视角相机位姿间距 {best_d:.2f} mm")
    if best_d > 20.0:
        notes.append("！警告：两视角位姿间距偏大，标定结果可能不可靠")
    return best1, notes


# --------------------------------------------------------------------- 输出


def rotation_to_xyaxes(R_cb: np.ndarray) -> str:
    """旋转矩阵 → MJCF xyaxes（相机 x 轴与 y 轴在基座系里的分量）。

    坐标约定不同，**不能直接取两列**：
      OpenCV 相机系：x 右、y 下、z 前（看向 +z）
      MuJoCo 相机系：x 右、y 上、z 后（看向 -z）
    对应关系是 x 相同、y 与 z 都取负。所以 mj_x = cv_x、mj_y = -cv_y，
    此时 mj_z = mj_x × mj_y = -cv_z，正好让「MuJoCo 看向 -z」等价于看向 cv_z。

    实测：若直接取两列，生成的相机光轴与正确方向恰好差 180.0°（朝上）。
    """
    return " ".join(f"{v:.6f}"
                    for v in np.concatenate([R_cb[:, 0], -R_cb[:, 1]]))


def format_camera_xml(R_cb: np.ndarray, t_cb: np.ndarray,
                      name: str = "hf868_camera") -> str:
    pos = " ".join(f"{v:.6f}" for v in t_cb)
    return (f'<camera name="{name}" pos="{pos}"\n'
            f'        xyaxes="{rotation_to_xyaxes(R_cb)}"/>')


def euler_deg(R_cb: np.ndarray) -> tuple[float, float]:
    """光轴的俯仰与偏航（度），便于人工核对。"""
    fwd = R_cb[:, 2]
    pitch = float(np.degrees(np.arcsin(np.clip(-fwd[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    return pitch, yaw


# --------------------------------------------------------------------- 抓图


def capture_image(device: str, frames: int = 12, warmup: int = 20) -> np.ndarray:
    """抓一张可用的图。房间偏暗时用长曝光，并在多帧里挑最亮的一张。"""
    import subprocess

    # 走 v4l2-ctl 设手动曝光：这台相机用 OpenCV 的 CAP_PROP_AUTO_EXPOSURE 会停流
    # 现场实测：gain=60/exp=3000 时 p95 只有 ~60，棋盘格检不出；gain=255/exp=5000
    # 时 p95 到 250，稳定检出 70 点。所以默认用能工作的组合。
    for key, val in (("auto_exposure", "1"), ("exposure_time_absolute", "5000"),
                     ("gain", "255")):
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{key}={val}"],
                       capture_output=True, check=False)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(warmup):
        cap.read()

    best, best_score = None, -1.0
    for _ in range(frames):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = float(np.percentile(g, 95))        # 高光分位比均值抗噪
        if score > best_score:
            best, best_score = f, score
    cap.release()

    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", "auto_exposure=3"],
                   capture_output=True, check=False)

    if best is None:
        raise RuntimeError("没有抓到帧")
    print(f"抓图完成：p95 亮度 {best_score:.0f}/255")
    if best_score < 60:
        print("  提示：画面偏暗。检测不到棋盘格时请补光后重试")
    return best


# --------------------------------------------------------------------- 自检


def make_synthetic(R_cb: np.ndarray, t_cb: np.ndarray, square_m: float,
                   K: np.ndarray, dist: np.ndarray, board: BoardPose,
                   cols: int = COLS, rows: int = ROWS,
                   out_w: int = CAM_W, out_h: int = CAM_H) -> np.ndarray:
    """合成一张「已知位姿下拍到的棋盘格」图像。"""
    obj = board_object_points(square_m, cols, rows)
    X_base = obj @ board.R.T + board.t
    rvec, _ = cv2.Rodrigues(R_cb.T)
    tvec = (-R_cb.T @ t_cb).reshape(3, 1)
    proj, _ = cv2.projectPoints(X_base, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    # 由投影结果反推单应，把标准棋盘图 warp 进画布，得到能被真实检测器识别的图
    H, _ = cv2.findHomography(obj[:, :2], proj)
    sq_px = 120                                   # 每格 120 px，保证 warp 后不糊
    ppm = sq_px / square_m
    canon = np.full(((rows + 1) * sq_px, (cols + 1) * sq_px), 255, np.uint8)
    for b in range(rows + 1):
        for a in range(cols + 1):
            if (a + b) % 2 == 0:
                canon[b * sq_px:(b + 1) * sq_px, a * sq_px:(a + 1) * sq_px] = 0

    # canon 像素 (0,0) 对应板坐标 (-s, -s)
    M = np.array([[ppm, 0.0, ppm * square_m],
                  [0.0, ppm, ppm * square_m],
                  [0.0, 0.0, 1.0]])
    img = cv2.warpPerspective(canon, H @ np.linalg.inv(M), (out_w, out_h),
                              borderValue=90)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def run_selftest() -> int:
    """合成数据自检：验证 PnP、对称性枚举、二义性破除的完整链路。"""
    print("=== 外参标定自检（合成数据，无需相机）===")
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])

    # scene_fixed.xml 里 front_camera 的位姿，转成 OpenCV 相机约定（x右/y下/z前）。
    # MuJoCo 相机轴是 x右/y上/z后，所以后两列取反；已用渲染实测核对：
    # 投影方块八角得 u[301,338] v[410,462]，与 MuJoCo 渲染 u[302,337] v[411,461] 一致。
    R_true = np.array([[0.0, 0.4472135955, -0.8944271910],
                       [1.0, 0.0, 0.0],
                       [0.0, -0.8944271910, -0.4472135955]])
    t_true = np.array([0.75, 0.0, 0.45])
    square_m = 0.030
    board = BoardPose(x0=0.30, y0=0.02, z0=0.002, yaw=np.radians(8.0))
    obj = board_object_points(square_m)
    X_base = obj @ board.R.T + board.t

    rvec, _ = cv2.Rodrigues(R_true.T)
    tvec = (-R_true.T @ t_true).reshape(3, 1)
    proj, _ = cv2.projectPoints(X_base, rvec, tvec, K, DIST)
    proj = proj.reshape(-1, 2)

    measured = X_base[[0, COLS - 1, (ROWS - 1) * COLS], :2]   # P0, P1, P2

    def run(pts_px, board_z=board.z0, arm_side="right"):
        bp, checks_ = board_pose_from_points(measured, square_m, COLS, ROWS, board_z)
        for w in layout_warnings(checks_):
            print(f"    ! {w}")
        cands = build_candidates(bp, square_m, pts_px, K, DIST)
        best, notes = choose_candidate(cands, arm_side, CAM_W)
        return best, bp, cands, notes

    # 基座原点在真值相机下的像素位置，用于确认 arm-side 该选哪边
    rv, _ = cv2.Rodrigues(R_true.T)
    tv = (-R_true.T @ t_true).reshape(3, 1)
    base_px_true, _ = cv2.projectPoints(np.zeros((1, 3)), rv, tv, K, DIST)
    side = "left" if base_px_true.reshape(2)[0] < CAM_W / 2 else "right"
    print(f"  真值下基座投影在 x={base_px_true.reshape(2)[0]:.0f}px -> 臂在画面{side}侧")

    # --- 第一段：理想投影，应精确复原 ---
    best, bp, cands, _ = run(proj, arm_side=side)
    te = float(np.linalg.norm(best["t_cb"] - t_true) * 1000)
    ae = float(np.degrees(np.arccos(
        np.clip((np.trace(best["R_cb"].T @ R_true) - 1) / 2, -1.0, 1.0))))
    print(f"  理想投影：位置误差 {te:.4f} mm，姿态误差 {ae:.5f}°，"
          f"残差 {best['resid_px']:.4f} px，选中={best['name']}")
    by_name = {c["name"]: c for c in cands}
    assert by_name["恒等"]["resid_px"] < 1e-6, "恒等解残差应接近 0"
    assert "180°" in by_name, "180° 候选应在枚举中"
    assert by_name["180°"]["resid_px"] < 1e-6, (
        "180° 解的残差也应接近 0 —— 这正是固有二义性的体现")
    assert abs(by_name["180°"]["t_cb"][0] - t_true[0]) > 0.1, (
        "180° 解应给出明显不同的相机位置")
    assert te < 0.5, f"理想情况下位置误差应接近 0，实际 {te:.3f} mm"
    assert ae < 0.05, f"理想情况下姿态误差应接近 0，实际 {ae:.4f}°"
    assert best["name"] == "恒等", f"应选中恒等解，实际 {best['name']}"

    # --- 第二段：加像素噪声，量化误差放大 ---
    rng = np.random.default_rng(20261020)
    print("  加 0.5 px 高斯噪声（5 次）：")
    worst_t = worst_a = 0.0
    for trial in range(5):
        noisy = proj + rng.normal(0.0, 0.5, proj.shape)
        best, bp, cands, _ = run(noisy, arm_side=side)
        te = float(np.linalg.norm(best["t_cb"] - t_true) * 1000)
        ae = float(np.degrees(np.arccos(
            np.clip((np.trace(best["R_cb"].T @ R_true) - 1) / 2, -1.0, 1.0))))
        worst_t, worst_a = max(worst_t, te), max(worst_a, ae)
        print(f"    trial {trial}: 位置 {te:5.2f} mm  姿态 {ae:.3f}°  "
              f"选中={best['name']}  残差 {best['resid_px']:.2f} px")
    assert worst_t < 8.0, f"噪声下位置误差过大 {worst_t:.2f} mm"
    assert worst_a < 0.8, f"噪声下姿态误差过大 {worst_a:.3f}°"

    # --- 第三段：量测有误差时的稳健性（模拟卷尺量偏 2 mm）---
    print("  标记点量测偏 2 mm：")
    for dx in (0.002, -0.002):
        biased = measured + np.array([dx, 0.0])
        bp, _ = board_pose_from_points(biased, square_m, COLS, ROWS, board.z0)
        cands = build_candidates(bp, square_m, proj, K, DIST)
        best, _ = choose_candidate(cands, side, CAM_W)
        te = float(np.linalg.norm(best["t_cb"] - t_true) * 1000)
        print(f"    P0 偏移 {dx*1000:+.0f} mm -> 位置误差 {te:5.2f} mm")
        assert te < 15.0, f"量测偏差 2 mm 时误差过大 {te:.2f} mm"

    # --- 第三段半：双视角一致性 ---
    # 单视角下基座投影恒在 x≈319px（贴近画面中心），arm-side 判据不可靠。
    # 这里证明换个板的摆放再解一次，就能自动锁定正确对称名。
    print("  双视角一致性（板移动后正确解不变）：")
    # 重新用无偏的 measured/proj 解一次，避免复用上面被偏移过的变量
    bp1, _ = board_pose_from_points(measured, square_m, COLS, ROWS, board.z0)
    cands1 = build_candidates(bp1, square_m, proj, K, DIST)
    board2 = BoardPose(x0=0.12, y0=0.18, z0=0.002, yaw=np.radians(-24.0))
    X2 = obj @ board2.R.T + board2.t
    proj2, _ = cv2.projectPoints(X2, rvec, tvec, K, DIST)
    proj2 = proj2.reshape(-1, 2)
    measured2 = X2[[0, COLS - 1, (ROWS - 1) * COLS], :2]
    bp2, _ = board_pose_from_points(measured2, square_m, COLS, ROWS, board2.z0)
    cands2 = build_candidates(bp2, square_m, proj2, K, DIST)

    best2, _ = resolve_two_views([cands1, cands2])
    te2 = float(np.linalg.norm(best2["t_cb"] - t_true) * 1000)
    print(f"    选定「{best2['name']}」，位置误差 {te2:.3f} mm")
    for c in cands2:
        if c["name"] == "180°":
            te_wrong = float(np.linalg.norm(c["t_cb"] - t_true) * 1000)
            break
    print(f"    （若误选 180° 会是 {te_wrong:.0f} mm 误差）")
    assert best2["name"] == "恒等", f"双视角应锁定恒等解，实际 {best2['name']}"
    assert te2 < 10.0, f"双视角位置误差过大 {te2:.2f} mm"

    # --- 第三段半之二：只量 2 个点也能解，且能自动识别沿长边/短边 ---
    print("  只量 2 个点（不提供 P2）：")
    # 长边（沿 cols，9 格 = 270 mm），短边（沿 rows，6 格 = 180 mm）
    # 注意：默认 COLS=7 < ROWS=10，所以 cols 轴是短边、rows 轴是长边
    for label, idx in (("沿 cols 轴（短边）", (0, COLS - 1)),
                       ("沿 rows 轴（长边）", (0, (ROWS - 1) * COLS))):
        m2 = X_base[list(idx), :2]
        bp2, chk = board_pose_from_points(m2, square_m, COLS, ROWS, board.z0)
        for w in layout_warnings(chk):
            print(f"      ! {w}")
        cds = build_candidates(bp2, square_m, proj, K, DIST)
        bs, _ = choose_candidate(cds, side, CAM_W)
        te = float(np.linalg.norm(bs["t_cb"] - t_true) * 1000)
        print(f"    {label}: 判定沿{chk['p0_p1_along']}, "
              f"实测 {chk['p0_p1_mm']:.0f}mm, 位置误差 {te:.2f} mm")
        assert te < 1.0, f"2 点标定误差过大 {te:.2f} mm"

    # --- 第四段：检测链路（合成棋盘图，验证检测器）---
    print("  检测链路（合成棋盘图）：")
    img = make_synthetic(R_true, t_true, square_m, K, DIST, board)
    det = detect_corners(img)
    if det is None:
        print("    合成图上检测失败")
        return 1
    print(f"    检测到 {len(det)} 个内角点（期望 {len(obj)}）")
    assert len(det) == len(obj), "内角点数量不符"

    print("\n自检通过。")
    return 0


# --------------------------------------------------------------------- 主流程


def parse_points(text: str) -> np.ndarray:
    """解析 "x1,y1 x2,y2 x3,y3"（毫米）为米制数组。"""
    pts = []
    for chunk in text.replace(";", " ").split():
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"标记点格式错误: {chunk}（应为 x,y）")
        pts.append([float(parts[0]) / 1000.0, float(parts[1]) / 1000.0])
    if len(pts) not in (2, 3):
        raise ValueError(f"需要 2 或 3 个标记点，收到 {len(pts)} 个")
    return np.array(pts, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HF868 外参标定（相机在机械臂基座坐标系里的位姿）",
        epilog=(
            "示例：\n"
            "  --selftest\n"
            "  --image photo.png --square-mm 25 "
            "--points '180,120 420,120 180,400' --arm-side left\n\n"
            "--points 三点顺序：\n"
            "  P0 = 板坐标原点的内角点\n"
            "  P1 = 从 P0 沿板 +X 方向走 (7-1)=6 格到达的内角点\n"
            "  P2 = 从 P0 沿板 +Y 方向走 (10-1)=9 格到达的内角点\n"
            "  坐标以基座轴心为原点，+x 为全零位手臂指向，+y 为其左侧，单位 mm。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="合成数据自检，不需要相机")
    ap.add_argument("--image", type=str, help="棋盘格照片（推荐 640x480 原图）")
    ap.add_argument("--capture", type=str, help="设备节点，直接抓图，如 /dev/video0")
    ap.add_argument("--square-mm", type=float, help="方格边长（毫米），必须实测")
    ap.add_argument("--points", type=str, help="三个标记内角点 (x,y)，单位 mm")
    ap.add_argument("--image2", type=str,
                    help="第二张照片（板换个位置再拍），用于自动破 180° 二义性")
    ap.add_argument("--points2", type=str,
                    help="第二张照片对应的三个标记内角点 (x,y)，单位 mm")
    ap.add_argument("--arm-side", choices=("left", "right"),
                    help="兜底：照片里机械臂在画面哪一侧（单视角时才用，可靠性低）")
    ap.add_argument("--board-z-mm", type=float, default=0.0,
                    help="内角点平面高度（板厚），默认 0")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--no-clahe", action="store_true", help="关闭对比度增强")
    ap.add_argument("--save-debug", type=str, help="把检测结果画到该路径便于核对")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()

    if args.square_mm is None or args.points is None:
        ap.error("需要 --square-mm 和 --points")
    if not args.image and not args.capture:
        ap.error("需要 --image 或 --capture")

    square_m = args.square_mm / 1000.0

    # 收集视角：每个视角 = (图像, 三个量测点, 标签)
    views = [("视角1", args.image, args.capture, args.points)]
    if args.image2 or args.points2:
        if not (args.image2 and args.points2):
            ap.error("--image2 与 --points2 需要同时提供")
        views.append(("视角2", args.image2, None, args.points2))

    # 板子平放桌面，+X × +Y 必须 = +z。若量测点的轴归属构成左手系，
    # 说明 P1/P2 与 cols/rows 的对应搞反了，这里统一纠正。
    norm_views = []
    for label, img_path, dev, pts_text in views:
        m = parse_points(pts_text)
        m2, nc, nr, info = normalize_handedness(m, args.cols, args.rows)
        if info.get("swapped"):
            print(f"\n{label}：检测到左手系（cross_z={info['cross_z']:+.3f}），"
                  f"已交换 P1/P2 且 cols/rows 由 "
                  f"{args.cols}x{args.rows} -> {nc}x{nr}")
            print("  说明：平放桌面的板子 +X×+Y 必须等于 +z；左手系说明量测点"
                  "的轴归属与实物相反，已自动纠正。")
        norm_views.append((label, img_path, dev, m2, nc, nr))

    view_cands: list[list[dict]] = []
    view_info: list[dict] = []
    for label, img_path, dev, measured, n_cols, n_rows in norm_views:
        image = (capture_image(dev) if dev else cv2.imread(img_path))
        if image is None:
            print(f"读不到图像: {img_path or dev}")
            return 1
        h, w = image.shape[:2]
        print(f"\n=== {label}：图像 {w}x{h} ===")

        sx, sy = w / CAM_W, h / CAM_H
        K = np.array([[FX * sx, 0, CX * sx], [0, FY * sy, CY * sy], [0, 0, 1.0]])
        if abs(sx - 1.0) > 1e-6:
            print(f"  注意：图像不是 640x480，内参已按 {sx:.4f} 缩放")

        corners = detect_corners(image, n_cols, n_rows,
                                 clahe=not args.no_clahe)
        if corners is None:
            print("  检测不到棋盘格。可尝试：补光 / 去掉 --no-clahe / "
                  f"确认内角点数是 {n_cols}x{n_rows}")
            return 2
        # 检测器可能用了转置后的网格尺寸（板子被转 90° 摆放）。obj 网格必须
        # 与检测时用的尺寸一致，否则点会错配，解出的位姿会整体差 90°。
        used_size = getattr(detect_corners, "last", {}).get(
            "size", (n_cols, n_rows))
        vc, vr = used_size
        if (vc, vr) != (n_cols, n_rows):
            print(f"  ！板子在画面里旋转了 90°，检测按 {vc}x{vr} 进行"
                  f"（你指定 {n_cols}x{n_rows}）。"
                  "请确认 P1/P2 的『沿X/沿Y』方向与之一致。")
        print(f"  检测到 {len(corners)} 个内角点")
        if args.save_debug:
            tag = label.replace("视角", "v")
            out = args.save_debug.replace(".png", f"_{tag}.png")
            vis = image.copy()
            for px, py in corners:
                cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
            cv2.putText(vis, f"{len(corners)} corners", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(out, vis)
            print(f"  检测结果已存 {out}（建议先看一眼角点是否齐全）")

        board, checks = board_pose_from_points(
            measured, square_m, vc, vr, args.board_z_mm / 1000.0)
        print(f"  板原点 ({board.x0*1000:.1f}, {board.y0*1000:.1f}) mm，"
              f"偏航 {np.degrees(board.yaw):.2f}°")
        # 两条边各自反推的方格边长：若相差明显，说明是量测误差而非方格尺寸不对
        s_x = checks["p0_p1_mm"] / (vc - 1)
        if "p0_p2_mm" in checks:
            s_y = checks["p0_p2_mm"] / (vr - 1)
            print(f"  反推方格边长：沿X {s_x:.2f} mm/格，沿Y {s_y:.2f} mm/格"
                  f"（标称 {args.square_mm}）")
            if abs(s_x - s_y) > 0.02 * args.square_mm:
                print(f"  ！两条边推出 {s_x:.2f} 与 {s_y:.2f} mm/格，"
                      f"相差 {abs(s_x-s_y)/args.square_mm*100:.1f}% —— "
                      "单一方格边长无法同时满足，属量测误差")
                print(f"    影响：尺度误差约 ±{abs(s_x-s_y)/args.square_mm*50:.1f}%，"
                      "会线性传给相机距离")
        print(f"  P0→P1 {checks['p0_p1_mm']:.1f} mm，沿{checks['p0_p1_along']}"
              f"（应 {checks['expected_along_x_mm'] if 'cols' in checks['p0_p1_along'] else checks['expected_along_y_mm']:.1f} mm）")
        if "orthogonality_err_deg" in checks:
            print(f"  P0→P2 {checks['p0_p2_mm']:.1f} mm（应 "
                  f"{checks['p0_p2_expected_mm']:.1f}），"
                  f"垂直度 {checks['orthogonality_err_deg']:+.2f}°")
        for msg in layout_warnings(checks):
            print(f"  ！{msg}")

        cands = build_candidates(board, square_m, corners, K, DIST, vc, vr)
        print("  各对称候选：")
        for c in cands:
            mark = "合理" if c["plausible"] else f"剔除（{c['reason']}）"
            print(f"    {c['name']:6s} 残差 {c['resid_px']:8.3f}px  "
                  f"相机 ({c['t_cb'][0]:+.3f},{c['t_cb'][1]:+.3f},{c['t_cb'][2]:+.3f})  "
                  f"{mark}")
        view_cands.append(cands)
        view_info.append({"label": label, "board": board, "checks": checks,
                          "image": img_path or dev, "w": w, "h": h,
                          "corners": corners, "K": K, "measured": measured,
                          "vc": vc, "vr": vr})

    # ---- 选解 ----
    if len(view_cands) >= 2:
        best, notes = resolve_two_views(view_cands)
        # 交叉验证：用选定的位姿在**每个**视角上各算一次残差。
        # 位姿只由一致性选出、没有被任一视角单独拟合，所以这是真正的交叉验证。
        reps = []
        for vi, cands_i in zip(view_info, view_cands):
            obj = board_object_points(square_m, vi["vc"], vi["vr"])
            X_i = obj @ vi["board"].R.T + vi["board"].t
            reps.append(reprojection_report(best["R_cb"], best["t_cb"], vi["K"],
                                            DIST, X_i, vi["corners"]))
            print(f"  {vi['label']} 交叉验证残差：mean {reps[-1]['mean_px']:.3f} px  "
                  f"max {reps[-1]['max_px']:.3f} px")
        rep = {
            "mean_px": float(np.mean([r["mean_px"] for r in reps])),
            "p95_px": float(np.max([r["p95_px"] for r in reps])),
            "max_px": float(np.max([r["max_px"] for r in reps])),
            "per_view": reps,
        }
    else:
        best, notes = choose_candidate(view_cands[0], args.arm_side,
                                       view_info[0]["w"])
        rep = best["report"]

    print("\n=== 选解 ===")
    for n in notes:
        print(f"  {n}")

    R_cb, t_cb = best["R_cb"], best["t_cb"]
    pitch, yaw = euler_deg(R_cb)

    print("\n=== 相机在基座坐标系里 ===")
    print(f"  位置 ({t_cb[0]:.4f}, {t_cb[1]:.4f}, {t_cb[2]:.4f}) m")
    print(f"  光轴俯仰 {pitch:.2f}°  偏航 {yaw:.2f}°")
    print(f"  距基座水平距离 {np.linalg.norm(t_cb[:2]):.4f} m")
    if rep["mean_px"] == rep["mean_px"]:      # 非 nan
        print(f"  复投影 mean {rep['mean_px']:.3f} px  p95 {rep['p95_px']:.3f} px  "
              f"max {rep['max_px']:.3f} px")
        if rep["mean_px"] > 2.0:
            print("  ！残差偏大：检查方格边长、量测点、图像清晰度")

    print("\n=== 与仿真 front_camera 对照 ===")
    print("  front_camera: pos (0.7500, 0.0000, 0.4500)  俯仰 26.57°")
    d = t_cb - np.array([0.75, 0.0, 0.45])
    print(f"  位置差 ({d[0]*1000:+.1f}, {d[1]*1000:+.1f}, {d[2]*1000:+.1f}) mm")

    xml = format_camera_xml(R_cb, t_cb)
    print("\n=== 粘进 scene_fixed.xml 的相机定义 ===")
    print(xml)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        "view1_board": {
            "x0_m": view_info[0]["board"].x0,
            "y0_m": view_info[0]["board"].y0,
            "z0_m": view_info[0]["board"].z0,
            "yaw_deg": float(np.degrees(view_info[0]["board"].yaw))},
        "camera_pos_m": t_cb.tolist(),
        "camera_R_cb": R_cb.tolist(),
        "camera_pitch_deg": pitch,
        "camera_yaw_deg": yaw,
        "reprojection": rep,
        "view1_layout_checks": view_info[0]["checks"],
        "n_views": len(view_cands),
        "square_mm": args.square_mm,
        "view1_measured_points_m": view_info[0].get("points", []),
        "chosen_symmetry": best["name"],
        "view1_candidates": [
            {"name": c["name"], "plausible": c["plausible"], "reason": c["reason"],
             "resid_px": c["resid_px"], "t_cb": c["t_cb"].tolist()}
            for c in view_cands[0]],
        "mjcf_camera": xml,
        "source_image": args.image or args.capture,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
