# 相机标定（真实机械臂）

实体 SO-101 使用的相机的内参，与 `../robots/so_follower/` 下的电机标定并列。

## hf868_135_640x480 —— 机械臂顶部相机

原为叉车上拍摄托盘二维码的相机（`forklift_pallet_cam.launch`，
设备 `/dev/forklift_artag_camera`，USB `05a3:9240`），现装在机械臂上作顶部相机。

| 项 | 值 |
|---|---|
| 分辨率 | 640×480 |
| fx, fy | 326.68129, 326.27539 |
| cx, cy | 319.20034, 248.66437 |
| 畸变 plumb_bob | k1=-0.012865 k2=-0.018761 p1=0.002971 p2=-0.001267 k3=0 |
| 视场角 | 88.82° × 72.67°（135° 广角镜头）|
| 重投影 RMS | 0.55 px |
| 标定日期 | 2025-06-05（71 个视角）|

## 两个文件

- `ost.yaml` —— ROS `camera_info_url` 原始格式，`usb_cam` 直接消费；
  与 `~/sfy_forklift_ros/.../HF868_135/640_480/ost.yaml` 逐字节相同。
- `intrinsics.yaml` —— 标注版，多出来的是溯源、质量判定和注意事项，供 Python 读取。
  两份数值必须一致。

## 怎么用

```python
# 从 so_arm_ws 目录运行；Python 环境借用 so_ws
#   ../so_ws/.venv/bin/python
from so_ws.common.camera_model import load_camera
cam = load_camera("hf868_135_640x480")     # 自动优先读本目录
cam.project(points_cam)                     # 相机系 3D -> 像素（含畸变）
cam.unproject(pixels, depth_m)              # 像素 + 深度 -> 相机系 3D
```

加载器按 `$SO_ARM_WS` → 当前目录 → `~/so_projects/so_arm_ws` → 仓库副本 的顺序查找，
所以从本目录启动时会优先用这里的标定。

## 注意

- 本目录是**真机侧权威副本**；`so_ws/references/project_metadata/configs/camera/`
  下还有一份版本控制镜像，供没有 so_arm_ws 的开发机使用。
  两边一致性由 `so_ws/tests/test_camera_model.py` 检查，漂移会测试失败。
- **没有外参**。像素到机械臂基座坐标仍需手眼标定。
- 只有 640×480 一组。要跑 1280×720 必须重新标定。
- udev 的 `/dev/forklift_artag_camera` 当前指向 `video1`（metadata 节点，取图报
  `VIDIOC_G_FMT: failed`），直接用 `/dev/video0`。
