"""先录帧再离线分析：采集阶段不做任何检测，保证帧率。

用法：
    python3 record_then_analyse.py record --seconds 40 --dir quick_id
    python3 record_then_analyse.py analyse --dir quick_id
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480
COLS, ROWS = 7, 10
SQUARE = 0.025

CANDIDATES = {
    "HF868_135 (ost.yaml)": 326.68,
    "HF868_90 (ost.yaml)": 508.96,
    "HF868_90_2 (ost.yaml)": 522.71,
    "HF868_90 (re-solve)": 496.8,
    "HF868_90_2 (re-solve)": 501.9,
}


def record(args):
    out = Path(args.dir)
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for p in fdir.glob("*.jpg"):
        p.unlink()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(1.0)

    t0 = time.time()
    i = 0
    last_log = 0.0
    print(f"录制中 {args.seconds}s，请慢慢移动/倾斜棋盘格，覆盖画面各区域...", flush=True)
    while time.time() - t0 < args.seconds:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        if i % args.every == 0:
            cv2.imwrite(str(fdir / f"{i//args.every:05d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        i += 1
        if time.time() - t0 - last_log > 5:
            last_log = time.time() - t0
            print(f"  [{last_log:4.1f}s] 已录 {i} 帧", flush=True)
    cap.release()
    saved = len(list(fdir.glob("*.jpg")))
    print(f"录制结束：读取 {i} 帧，保存 {saved} 张到 {fdir}")
    return 0


def detect(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit).reshape(-1, 2)


def analyse(args):
    out = Path(args.dir)
    files = sorted((out / "frames").glob("*.jpg"))
    print(f"离线分析 {len(files)} 张")
    if not files:
        print("FAILED 没有图片")
        return 2

    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE

    objpoints, imgpoints, metas = [], [], []
    prev = None
    t0 = time.time()
    for k, p in enumerate(files):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        c = detect(g)
        if c is None:
            continue
        center = c.mean(axis=0)
        span = float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1])))
        if span < 120:
            continue
        if prev is not None:
            moved = np.linalg.norm(center - prev[0]) > 12
            resized = abs(span - prev[1]) / prev[1] > 0.05
            if not (moved or resized):
                continue
        prev = (center, span)
        objpoints.append(objp.copy())
        imgpoints.append(c.astype(np.float32))
        metas.append({"file": p.name, "span": round(span, 1),
                      "mean": round(float(g.mean()), 1)})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(files)}  已收 {len(objpoints)} 个有效视角", flush=True)
    print(f"分析耗时 {time.time()-t0:.1f}s，有效视角 {len(objpoints)}")

    if len(objpoints) < 5:
        print("FAILED 有效视角不足 5 个")
        (out / "analyse.json").write_text(json.dumps({"error": "too few views",
                                                      "views": len(objpoints)}, indent=2))
        return 2

    rms, K, D, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (W, H), None, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"\n解出 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} RMS={rms:.3f}px")

    print("\n=== 与历史标定比对 ===")
    dist = {n: abs(fx - v) / v * 100 for n, v in CANDIDATES.items()}
    best = min(dist, key=dist.get)
    for n, d in sorted(dist.items(), key=lambda kv: kv[1]):
        print(f"  {n:24s} fx={CANDIDATES[n]:7.2f}  偏差 {d:5.1f}%"
              + ("   <== 最接近" if n == best else ""))

    res = {"resolution": [W, H], "pattern": [COLS, ROWS], "views": len(objpoints),
           "fx": round(float(fx), 3), "fy": round(float(fy), 3),
           "cx": round(float(cx), 3), "cy": round(float(cy), 3),
           "rms_px": round(float(rms), 4), "D": D.ravel()[:5].tolist(),
           "deviation_pct": {n: round(d, 2) for n, d in dist.items()},
           "closest": best, "metas": metas}
    (out / "analyse.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n已存 {out}/analyse.json\n=> 判定最接近：{best}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--dir", default="quick_id")
    r.add_argument("--seconds", type=float, default=40)
    r.add_argument("--every", type=int, default=2)
    r.set_defaults(func=record)
    a = sub.add_parser("analyse")
    a.add_argument("--dir", default="quick_id")
    a.set_defaults(func=analyse)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
