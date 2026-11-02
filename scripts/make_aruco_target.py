"""生成腕部标定靶标（ArUco 50 mm）的精确尺寸 PDF。

为什么用 ArUco 而不是小棋盘格
----------------------------
ArUco 有唯一 ID，**不存在 180°/镜像二义性**，单张图就能给出完整 6DOF 位姿。
小棋盘格仍是对称靶标，每张图仍有两个等价解，会污染手眼标定。

尺寸依据（实测自 MuJoCo 模型）
----------------------------
`wrist` 顶部可用平面只有 17.5 x 54 mm，装不下 60 mm 靶标，所以：
  - 黑色方块 50 mm（DICT_4X4_50 为 6x6 模块 -> 每模块 8.33 mm）
  - 白静默区 10 mm（>= 1 模块，ArUco 要求不可省略）
  - 总占位 70 mm，需要一块 70x70 mm 的**硬衬板**（2 mm 卡纸/塑料片）
在 350 mm 拍摄距离下，50 mm 靶标约 47 px 宽、7.8 px/模块，可可靠解码。
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

DPI = 300
MM_PER_INCH = 25.4

A4_MM = (210.0, 297.0)
MARKER_MM = 50.0          # 黑色方块总边长（含自带黑边框）
QUIET_MM = 10.0           # 白色静默区
CARD_MM = 70.0            # 建议的硬衬板边长 = MARKER + 2*QUIET
RULER_MM = 100.0          # 校准线

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_ID = 0


def mm_to_px(mm: float) -> int:
    return round(mm / MM_PER_INCH * DPI)


def make_page() -> Image.Image:
    w, h = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])
    arr = np.full((h, w), 255, np.uint8)

    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    side = mm_to_px(MARKER_MM)
    marker = cv2.aruco.generateImageMarker(adict, ARUCO_ID, side)
    marker = cv2.resize(marker, (side, side), interpolation=cv2.INTER_NEAREST)

    total = mm_to_px(CARD_MM)
    block = np.full((total, total), 255, np.uint8)
    off = mm_to_px(QUIET_MM)
    block[off:off + side, off:off + side] = marker
    # 虚线剪切线（即衬板 70 mm 边界）
    dash, gap = mm_to_px(4), mm_to_px(3)
    for t in range(0, total, dash + gap):
        block[0, t:t + dash] = 0
        block[-1, t:t + dash] = 0
        block[t:t + dash, 0] = 0
        block[t:t + dash, -1] = 0

    y0 = mm_to_px(34.0)
    x0 = (w - total) // 2
    arr[y0:y0 + total, x0:x0 + total] = block

    # 校准线
    ry = y0 + total + mm_to_px(24.0)
    rx0 = (w - mm_to_px(RULER_MM)) // 2
    rx1 = rx0 + mm_to_px(RULER_MM)
    cv2.line(arr, (rx0, ry), (rx1, ry), 0, 3)
    for x in (rx0, rx1):
        cv2.line(arr, (x, ry - mm_to_px(4)), (x, ry + mm_to_px(4)), 0, 3)
    for i in range(1, 10):
        x = rx0 + mm_to_px(RULER_MM * i / 10)
        cv2.line(arr, (x, ry - mm_to_px(2)), (x, ry + mm_to_px(2)), 0, 1)

    txt = [
        f"ArUco  DICT_4X4_50   ID = {ARUCO_ID}",
        f"BLACK SQUARE = {MARKER_MM:.0f} mm    CUT ALONG DASHED LINE = {CARD_MM:.0f} mm",
        f"WHITE QUIET ZONE = {QUIET_MM:.0f} mm  --  must stay white",
        f"RULER LINE = {RULER_MM:.0f} mm  (measure after printing)",
        "Print at 100% scale. Glue onto 2 mm rigid backing, tape to WRIST TOP.",
    ]
    ty = ry + mm_to_px(14.0)
    for i, t in enumerate(txt):
        cv2.putText(arr, t, (mm_to_px(18), ty + i * mm_to_px(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, 0, 1, cv2.LINE_AA)

    return Image.fromarray(arr)


def main() -> None:
    page = make_page()
    out_png = "results/extrinsics/aruco_target_50mm.png"
    out_pdf = "results/extrinsics/aruco_target_50mm.pdf"
    page.save(out_png, "PNG", dpi=(DPI, DPI))
    page.save(out_pdf, "PDF", resolution=float(DPI))
    print(f"已生成 {out_png}")
    print(f"已生成 {out_pdf}")
    print(f"黑色方块 {MARKER_MM:.0f} mm，静默区 {QUIET_MM:.0f} mm，"
          f"剪切线 {CARD_MM:.0f} mm")
    print(f"校准线 {RULER_MM:.0f} mm —— 打印后务必量一下")


if __name__ == "__main__":
    main()
