"""对照两套 cam_info 目录里的所有 ost.yaml，找命名错位和重复。"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict


def info(p):
    t = open(p, encoding="utf-8", errors="replace").read()
    num = lambda s: [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]
    K = num(re.search(r"camera_matrix:.*?data:\s*\[(.*?)\]", t, re.S).group(1))
    D = num(re.search(r"distortion_coefficients:.*?data:\s*\[(.*?)\]", t, re.S).group(1))
    w = int(re.search(r"image_width:\s*(\d+)", t).group(1))
    h = int(re.search(r"image_height:\s*(\d+)", t).group(1))
    return w, h, K, D


trees = {
    "~/cam_info": os.path.expanduser("~/cam_info"),
    "usb_cam/cam_info": os.path.expanduser(
        "~/sfy_forklift_ros/src/sfy_forklift_ros/usb_cam/cam_info"),
}

rows = []
for label, root in trees.items():
    for p in sorted(glob.glob(root + "/**/ost.yaml", recursive=True)):
        w, h, K, D = info(p)
        rel = p[len(root) + 1:].replace("/ost.yaml", "")
        rows.append({"tree": label, "name": rel, "w": w, "h": h,
                     "fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5], "D": D, "path": p})

hdr = "{:18s} {:24s} {:10s} {:>9s} {:>9s} {:>8s} {:>8s}".format(
    "tree", "name", "res", "fx", "fy", "cx", "cy")
print(hdr)
print("-" * len(hdr))
for r in rows:
    res = "{}x{}".format(r["w"], r["h"])
    print("{:18s} {:24s} {:10s} {:9.2f} {:9.2f} {:8.2f} {:8.2f}".format(
        r["tree"], r["name"], res, r["fx"], r["fy"], r["cx"], r["cy"]))

print("\n=== 同一组内参出现在多个名字下 ===")
sig = defaultdict(list)
for r in rows:
    key = (round(r["fx"], 2), round(r["fy"], 2), round(r["cx"], 2), round(r["cy"], 2),
           r["w"], r["h"])
    sig[key].append("{}/{}".format(r["tree"], r["name"]))

for k, v in sig.items():
    if len(v) > 1:
        print("  fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}  {}x{}".format(
            k[0], k[1], k[2], k[3], k[4], k[5]))
        for name in v:
            print("      " + name)

print("\n=== launch 实际引用的文件 ===")
launch_refs = [
    ("usb_cam-test.launch",
     os.path.expanduser("~/sfy_forklift_ros/src/sfy_forklift_ros/usb_cam/launch/usb_cam-test.launch")),
    ("ar_track_camera.launch",
     os.path.expanduser("~/sfy_forklift_ros/src/sfy_forklift_ros/usb_cam/launch/ar_track_camera.launch")),
]
for tag, lp in launch_refs:
    if not os.path.exists(lp):
        print("  {} : 文件不存在".format(tag))
        continue
    t = open(lp, encoding="utf-8", errors="replace").read()
    m = re.search(r'camera_info_url"\s+value="([^"]+)"', t)
    ref = m.group(1) if m else "(未设置)"
    print("  {:22s} -> {}".format(tag, ref))
    mm = re.search(r"/cam_info/(.+?)/ost\.yaml", ref)
    if mm:
        key = mm.group(1)
        target = os.path.expanduser(
            "~/sfy_forklift_ros/src/sfy_forklift_ros/usb_cam/cam_info/" + key + "/ost.yaml")
        if os.path.exists(target):
            w, h, K, D = info(target)
            print("     解析: {}  存在=True  {}x{}  fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}".format(
                key, w, h, K[0], K[4], K[2], K[5]))
            print("     D={}".format(D))
        else:
            print("     解析: {}  存在=False  <== 路径失效！".format(key))
