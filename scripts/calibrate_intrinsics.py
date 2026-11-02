#!/usr/bin/env python3
"""相机内参标定（采集 + 求解），只依赖 cv2/numpy，可直接在远端机器跑。

棋盘格规格通过 --cols/--rows 指定为**内角点数**。默认 8 列 x 6 行？不，默认按
本项目 PDF 实测结果：7x9 方格 -> 内角点 6x8。用 --cols 6 --rows 8。

方格物理边长只影响平移向量，不影响内参，所以 --square 取任意值都能得到正确的
fx/fy/cx/cy 与畸变系数（见 README 说明）。

用法：
    python3 calibrate_intrinsics.py capture --seconds 120 --out calib_640x480
    python3 calibrate_intrinsics.py solve --dir calib_640x480
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- 棋盘格检测

def make_object_points(cols: int, rows: int, square: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square
    return objp


def detect(gray: np.ndarray, cols: int, rows: int, fast: bool = False):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def board_structure(corners: np.ndarray, cols: int) -> dict:
    """描述棋盘在图像中的位置/大小/倾角，用于去重和覆盖度统计。"""
    pts = corners.reshape(-1, 2)
    center = pts.mean(axis=0)
    # 用相邻角点估计方块在图像中的平均边长
    d1 = np.linalg.norm(pts[1] - pts[0])
    d2 = np.linalg.norm(pts[cols] - pts[0]) if len(pts) > cols else d1
    sq = float((d1 + d2) / 2.0)
    # 倾角：网格主轴方向
    axis = pts[cols - 1] - pts[0]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return {"center": center, "square_px": sq, "angle": angle}


# ---------------------------------------------------------------- 覆盖度

def region_of(cx: float, cy: float, w: int, h: int) -> int:
    """把画面切成 3x3，返回格子编号 0..8。"""
    gx = min(2, max(0, int(cx / (w / 3.0))))
    gy = min(2, max(0, int(cy / (h / 3.0))))
    return gy * 3 + gx


def size_class(square_px: float, w: int) -> int:
    """按方块占画面宽度的比例分档，保证有远近变化。"""
    frac = square_px / w
    for i, thr in enumerate((0.04, 0.07, 0.11, 0.16)):
        if frac < thr:
            return i
    return 4


# ---------------------------------------------------------------- 采集

def cmd_capture(args) -> int:
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR 打不开相机 {args.device}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    all_corners: list[np.ndarray] = []
    seen_regions: set[int] = set()
    seen_sizes: set[int] = set()
    prev = None
    t0 = time.time()
    last_report = 0.0
    tries = 0

    print(f"采集中：{args.width}x{args.height}，最多 {args.max_views} 个视角 / "
          f"{args.seconds}s。请举着棋盘格连续移动、倾斜、远近变化。", flush=True)
    try:
        while True:
            elapsed = time.time() - t0
            if elapsed > args.seconds:
                print("到时间上限，停止采集")
                break
            if len(all_corners) >= args.max_views:
                print("已收满目标视角数，停止采集")
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                tries += 1
                if tries > 100:
                    print("ERROR 连续读帧失败")
                    break
                continue
            tries = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect(gray, args.cols, args.rows, fast=True)
            if corners is None:
                if elapsed - last_report > 5:
                    print(f"  [{elapsed:5.1f}s] 未检测到棋盘格。请让整块棋盘完整出现在画面里，"
                          f"占画面 1/3 以上。", flush=True)
                    last_report = elapsed
                continue

            info = board_structure(corners, args.cols)
            cx, cy = info["center"]
            h, w = gray.shape
            # 要求棋盘完整在画面内，留边
            if cx < 5 or cx > w - 5 or cy < 5 or cy > h - 5:
                continue
            # 方块太小说明太远，标定精度差
            if info["square_px"] < args.min_square_px:
                if elapsed - last_report > 5:
                    print(f"  [{elapsed:5.1f}s] 棋盘太小（方块 {info['square_px']:.1f}px，"
                          f"要求 ≥{args.min_square_px:.0f}px），请靠近一些。", flush=True)
                    last_report = elapsed
                continue

            # 去重：与上一个接受的视角相比，位移/大小/角度都要有变化
            if prev is not None:
                moved = np.linalg.norm(info["center"] - prev["center"]) > args.min_move_px
                resized = abs(info["square_px"] - prev["square_px"]) / prev["square_px"] > 0.08
                rotated = abs(info["angle"] - prev["angle"]) > 5.0
                if not (moved or resized or rotated):
                    continue

            idx = len(all_corners)
            all_corners.append(corners)
            seen_regions.add(region_of(cx, cy, w, h))
            seen_sizes.add(size_class(info["square_px"], w))
            prev = info
            cv2.imwrite(str(frames_dir / f"{idx:03d}.png"), frame)
            print(f"  [{elapsed:5.1f}s] ACCEPT {idx + 1:2d}  "
                  f"center=({cx:5.0f},{cy:5.0f}) square={info['square_px']:5.1f}px "
                  f"angle={info['angle']:6.1f}°  区域覆盖 {len(seen_regions)}/9 "
                  f"尺寸档 {len(seen_sizes)}/5", flush=True)
            last_report = elapsed
    finally:
        cap.release()

    if not all_corners:
        print("\nFAILED 没有采到任何有效视角。")
        return 2

    objp = make_object_points(args.cols, args.rows, args.square)
    np.savez(out / "corners.npz",
             imgpoints=np.array(all_corners, dtype=np.float32),
             objpoints=np.array([objp] * len(all_corners), dtype=np.float32),
             image_size=np.array([args.width, args.height]),
             cols=args.cols, rows=args.rows, square=args.square)

    print(f"\n采集完成：{len(all_corners)} 个视角，结果在 {out}/corners.npz")
    print(f"区域覆盖 {len(seen_regions)}/9，尺寸档 {len(seen_sizes)}/5")
    missing = sorted(set(range(9)) - seen_regions)
    if missing:
        print(f"WARNING 以下画面区域没有覆盖到（3x3 编号 0-8）：{missing}")
    if len(seen_sizes) < 3:
        print("WARNING 远近变化不足，建议补充更近和更远的视角")
    return 0


# ---------------------------------------------------------------- 求解

def solve(objpoints, imgpoints, size, cols, rows, square):
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None)
    return rms, K, D, rvecs, tvecs


def per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, D):
    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        errs.append(float(e))
    return np.array(errs)


def cmd_solve(args) -> int:
    d = Path(args.dir)
    data = np.load(d / "corners.npz", allow_pickle=True)
    imgp = list(data["imgpoints"])
    objp = list(data["objpoints"])
    size = tuple(int(v) for v in data["image_size"])
    cols, rows = int(data["cols"]), int(data["rows"])
    square = float(data["square"])
    w, h = size

    print(f"输入：{len(imgp)} 个视角，{w}x{h}，内角点 {cols}x{rows}，方格 {square}")
    rms, K, D, rvecs, tvecs = solve(objp, imgp, size, cols, rows, square)
    errs = per_view_errors(objp, imgp, rvecs, tvecs, K, D)
    print(f"初解 RMS={rms:.4f} px，单视角误差中位数={np.median(errs):.4f} "
          f"最大={errs.max():.4f}")

    # 剔除离群视角后重解
    thresh = max(1.0, 3.0 * float(np.median(errs)))
    keep = [i for i, e in enumerate(errs) if e <= thresh]
    if len(keep) < len(errs) and len(keep) >= 8:
        print(f"剔除 {len(errs) - len(keep)} 个误差 > {thresh:.3f}px 的视角后重解")
        objp2 = [objp[i] for i in keep]
        imgp2 = [imgp[i] for i in keep]
        rms, K, D, rvecs, tvecs = solve(objp2, imgp2, size, cols, rows, square)
        errs = per_view_errors(objp2, imgp2, rvecs, tvecs, K, D)
        print(f"重解 RMS={rms:.4f} px，视角数={len(keep)}")
        objp, imgp = objp2, imgp2

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # bootstrap：重采样视角，看内参的稳定程度
    rng = np.random.default_rng(0)
    boots = []
    n = len(objp)
    for _ in range(30):
        idxs = rng.integers(0, n, n)
        try:
            _, Kb, _, _, _ = cv2.calibrateCamera(
                [objp[i] for i in idxs], [imgp[i] for i in idxs], size, None, None)
        except cv2.error:
            continue
        boots.append([Kb[0, 0], Kb[1, 1], Kb[0, 2], Kb[1, 2]])
    boots = np.array(boots) if boots else np.zeros((0, 4))

    print("\n===== 标定结果 =====")
    print(f"分辨率      {w}x{h}")
    print(f"fx          {fx:.3f}")
    print(f"fy          {fy:.3f}")
    print(f"cx          {cx:.3f}")
    print(f"cy          {cy:.3f}")
    print(f"畸变 D(raw) {np.array2string(D.ravel(), precision=6)}")
    print(f"fx/fy 差异  {abs(fx - fy) / fy * 100:.2f} %")
    print(f"主点偏移    ({cx - (w / 2 - 0.5):+.1f}, {cy - (h / 2 - 0.5):+.1f}) px")
    print(f"水平 FOV    {2 * np.degrees(np.arctan((w / 2) / fx)):.2f}°")
    print(f"垂直 FOV    {2 * np.degrees(np.arctan((h / 2) / fy)):.2f}°")
    print(f"RMS 重投影  {rms:.4f} px  （<0.5 优秀, <1.0 可用, >1.5 偏差）")
    if len(boots) >= 5:
        std = boots.std(axis=0)
        print(f"bootstrap 标准差 fx={std[0]:.2f} fy={std[1]:.2f} "
              f"cx={std[2]:.2f} cy={std[3]:.2f}")

    # 质量判定
    verdict = []
    if rms > 1.5:
        verdict.append("RMS 偏大，标定不可靠")
    if abs(fx - fy) / fy > 0.02:
        verdict.append("fx/fy 差异 >2%，像元非方形或检测有误")
    if abs(cx - (w / 2 - 0.5)) > 0.1 * w or abs(cy - (h / 2 - 0.5)) > 0.1 * h:
        verdict.append("主点偏离中心 >10%，视角覆盖可能不均")
    if len(boots) >= 5 and boots[:, 0].std() > 0.02 * fx:
        verdict.append("bootstrap 显示 fx 不稳定（>2%）")
    print("\n结论：" + ("通过" if not verdict else "；".join(verdict)))

    result = {
        "resolution": [w, h],
        "intrinsics": {"fx": round(float(fx), 6), "fy": round(float(fy), 6),
                       "cx": round(float(cx), 6), "cy": round(float(cy), 6)},
        "camera_matrix": K.tolist(),
        "distortion_plumb_bob": D.ravel()[:5].tolist(),
        "rms_reprojection_px": round(float(rms), 5),
        "views_used": int(len(objp)),
        "fov_deg": {"horizontal": round(float(2 * np.degrees(np.arctan((w / 2) / fx))), 3),
                    "vertical": round(float(2 * np.degrees(np.arctan((h / 2) / fy))), 3)},
        "bootstrap_std": ({"fx": round(float(boots[:, 0].std()), 4),
                           "fy": round(float(boots[:, 1].std()), 4),
                           "cx": round(float(boots[:, 2].std()), 4),
                           "cy": round(float(boots[:, 3].std()), 4)}
                          if len(boots) >= 5 else None),
        "quality_issues": verdict,
        "per_view_rms_px": [round(float(e), 4) for e in errs],
    }
    (d / "intrinsics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    ost = f"""image_width: {w}
image_height: {h}
camera_name: {args.camera_name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{K[0,0]:.6f}, 0.000000, {K[0,2]:.6f},
         0.000000, {K[1,1]:.6f}, {K[1,2]:.6f},
         0.000000, 0.000000, 1.000000]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f'{v:.6f}' for v in D.ravel()[:5])}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1., 0., 0.,
         0., 1., 0.,
         0., 0., 1.]
projection_matrix:
  rows: 3
  cols: 4
  data: [{K[0,0]:.6f}, 0.000000, {K[0,2]:.6f}, 0.000000,
         0.000000, {K[1,1]:.6f}, {K[1,2]:.6f}, 0.000000,
         0.000000, 0.000000, 1.000000, 0.000000]
"""
    (d / "ost.yaml").write_text(ost)
    print(f"\n已写出 {d}/intrinsics.json 和 {d}/ost.yaml")
    return 0


# ---------------------------------------------------------------- CLI

# ---------------------------------------------------------------- 探测棋盘规格

def candidate_shapes(lo: int = 5, hi: int = 12) -> list[tuple[int, int]]:
    """所有可能的 (内角点列, 内角点行) 组合。长边优先试，命中更快。"""
    out = []
    for c in range(lo, hi + 1):
        for r in range(lo, hi + 1):
            out.append((c, r))
    out.sort(key=lambda cr: -(cr[0] * cr[1]))
    return out


def detect_any(gray: np.ndarray, shapes) -> list[tuple[int, int, np.ndarray]]:
    """在所有候选规格里找能检出的。SB 优先（鲁棒），失败再退回经典实现。"""
    hits = []
    for (c, r) in shapes:
        corners = None
        try:
            ok, c_ = cv2.findChessboardCornersSB(gray, (c, r))
            if ok:
                corners = c_.reshape(-1, 2)
        except cv2.error:
            pass
        if corners is None:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, c_ = cv2.findChessboardCorners(gray, (c, r), flags)
            if ok:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(
                    gray, c_, (11, 11), (-1, -1), criteria).reshape(-1, 2)
        if corners is not None:
            hits.append((c, r, corners))
    return hits


def cmd_probe(args) -> int:
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR 打不开相机 {args.device}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    shapes = candidate_shapes()
    votes: dict[tuple[int, int], int] = {}
    t0 = time.time()
    best_frame = None
    best_hit: dict = {}
    frame_i = 0
    print(f"探测棋盘规格：{args.seconds:.0f} 秒内请把整块棋盘格举进画面"
          f"（{len(shapes)} 种候选，每约 0.6 秒扫一次）。", flush=True)
    try:
        while time.time() - t0 < args.seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_i += 1
            # 全量扫描很贵，抽样做
            if frame_i % 12 != 1:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hits = detect_any(gray, shapes)
            for (c, r, corners) in hits:
                votes[(c, r)] = votes.get((c, r), 0) + 1
                area = float(np.ptp(corners[:, 0]) * np.ptp(corners[:, 1]))
                if best_frame is None or area > best_hit.get("area", 0.0):
                    best_frame, best_hit = frame.copy(), {
                        "cols": c, "rows": r, "area": area, "corners": corners}
            if hits:
                top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
                fmt = ", ".join(f"{c}x{r}({n})" for (c, r), n in top)
                print(f"  [{time.time() - t0:4.1f}s] 检出：{fmt}", flush=True)
    finally:
        cap.release()

    if not votes:
        print("\nFAILED 没有检出任何棋盘格。请确认棋盘完整在画面内、亮度足够。")
        return 2

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    print("\n===== 探测结果（按命中次数）=====")
    for (c, r), n in ranked[:6]:
        print(f"  内角点 {c:2d} x {r:2d}  命中 {n:3d} 次   （对应方格 {c + 1} x {r + 1}）")

    (bc, br), bn = ranked[0]
    print(f"\n=> 判定：内角点 {bc} x {br}，方格 {bc + 1} x {br + 1}")
    if len(ranked) > 1 and ranked[1][1] >= bn * 0.5:
        (c2, r2), n2 = ranked[1]
        print(f"   注意：{c2}x{r2} 也有 {n2} 次命中（{bn} vs {n2}）。"
              f"小规格是大规格的子集，故取点数更多者。")

    Path(args.dir).mkdir(parents=True, exist_ok=True)
    if best_frame is not None:
        vis = best_frame.copy()
        cv2.drawChessboardCorners(
            vis, (bc, br), best_hit["corners"].reshape(-1, 1, 2).astype(np.float32), True)
        cv2.imwrite(str(Path(args.dir) / "probe_pattern.png"), vis)
        cv2.imwrite(str(Path(args.dir) / "probe_raw.png"), best_frame)
        print(f"   检测示意已存 {Path(args.dir) / 'probe_pattern.png'}")

    (Path(args.dir) / "pattern.json").write_text(json.dumps(
        {"cols": bc, "rows": br,
         "votes": {f"{c}x{r}": n for (c, r), n in ranked[:8]}}, indent=2), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="相机内参标定")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="自动探测棋盘格规格")
    pr.add_argument("--dir", default="calib_run")
    pr.add_argument("--device", type=int, default=0)
    pr.add_argument("--width", type=int, default=640)
    pr.add_argument("--height", type=int, default=480)
    pr.add_argument("--seconds", type=float, default=25)
    pr.set_defaults(func=cmd_probe)

    c = sub.add_parser("capture", help="采集标定图")
    c.add_argument("--dir", default="calib_run")
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--cols", type=int, default=6, help="内角点列数")
    c.add_argument("--rows", type=int, default=8, help="内角点行数")
    c.add_argument("--square", type=float, default=0.033, help="方格边长(m)，不影响内参")
    c.add_argument("--seconds", type=float, default=120)
    c.add_argument("--max-views", type=int, default=40)
    c.add_argument("--min-square-px", type=float, default=14.0)
    c.add_argument("--min-move-px", type=float, default=18.0)
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve", help="求解内参")
    s.add_argument("--dir", default="calib_run")
    s.add_argument("--camera-name", default="artag_usb_camera")
    s.set_defaults(func=cmd_solve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
