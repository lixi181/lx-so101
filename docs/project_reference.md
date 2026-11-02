> 开发版说明：以下为历史技术记录，文中旧批次、远端环境与实验结果不随本仓库分发。当前安装和运行方式见 [项目首页](../README.md)。

# 项目技术参考与历史记录

本文于 2026-10-26 从根目录 README 拆出，保留相机标定、抓取判据、数据格式及历史批次的详细记录。
常用启动入口和当前本机状态请看 [项目首页](../README.md)，手柄与官方 IK 说明见
[手柄操作说明](gamepad_mujoco.md)。

历史设备节点、测量和成功率对应各自调查或实验当时，本次整理没有重新连接设备或复跑实验。
旧文中的 `raw_v1`、`raw_v1_pilot` 是历史批次；当前本机数据批次为 `raw_v2`、`raw_v3`。
文中的脚本命令和代码路径仍以项目根目录 `lx-so101` 为基准。

导航：[相机内参](#真实相机内参) · [相机外参](#相机外参provisional) ·
[抓放任务](#抓取盒内放置和返回初始姿态) · [数据采集](#本地数据集采集) ·
[格式转换](#lerobot-格式转换) · [试点批次](#第一版数据集-v1-的试点批次) ·
[分层采样](#分层采样) · [评估布局](#评估布局eval_layouts_v1)

## 真实相机内参

机械臂顶部相机（原为叉车拍摄托盘二维码的相机）的内参已经存档，无需重新标定。

**真机侧的权威副本在 `so_arm_ws`**：`<so_arm_ws>/calibration/cameras/hf868_135_640x480/`，
与电机标定 `calibration/robots/so_follower/` 并列。本仓库
`references/project_metadata/configs/camera/hf868_135_640x480/` 下是版本控制镜像，
供没有 `so_arm_ws` 的开发机使用；`tests/test_camera_model.py` 在真机侧副本存在时检查两边一致性。

```python
# 在 so_arm_ws 目录下运行（Python 环境借用 so_ws：../so_ws/.venv/bin/python）
from so_ws.common.camera_model import load_camera

cam = load_camera("hf868_135_640x480")
cam.fx, cam.fy, cam.cx, cam.cy      # 326.68129, 326.27539, 319.20034, 248.66437
cam.D                                # plumb_bob: k1 k2 p1 p2 k3
cam.hfov_deg, cam.vfov_deg           # 88.82°, 72.67°
cam.project(points_cam)              # 相机系 3D -> 像素（含畸变）
cam.unproject(pixels, depth_m)       # 像素 + 深度 -> 相机系 3D
cam.pixel_ray(pixel)                 # 像素 -> 单位方向向量
```

每个标定目录里有两份文件：

- `intrinsics.yaml`：标注版，含溯源、质量判定和注意事项，供本工作区读取；
- `ost.yaml`：ROS `usb_cam` 的 `camera_info_url` 原始格式，与上面数值一致。

加载器按 `$SO_ARM_WS` → 当前目录 → `~/so_projects/so_arm_ws` → 本仓库镜像 的顺序
查找 `<目录>/intrinsics.yaml`，所以从 `so_arm_ws` 启动时会优先用真机侧的标定。
默认相机登记在 `references/project_metadata/configs/cameras.yaml` 的 `cameras.top`。

几个使用要点：

- 这台相机是 **135° 广角**，`fx≈327`。叉车 `usb_cam/cam_info/HF868_90/640_480`（`fx=522.71`）
  是另一台机器的参数，切勿混用 —— 焦距差 60%，`ar_track_alvar` 算出的距离会系统性偏近约 1.6 倍。
- 只有 **640×480** 一组标定。`HF868_*` 下所有 `1280×720` 的历史标定包都无法检出角点，不可用；
  要在该分辨率下使用必须重新标定。
- `plumb_bob` 的 `k3` 为 0 是标定输出的原值，不是缺失。
- 内参不含**外参**。像素到机械臂基座坐标仍需手眼标定，`cameras.yaml` 里 `extrinsics: null`。
  外参目前有一份**临时估算值**写进了 MuJoCo，见下节「相机外参」。
- udev 的 `/dev/forklift_artag_camera` 目前指向 `video1`（metadata 节点，取图会报
  `VIDIOC_G_FMT: failed`），直接改用 `/dev/video0`；修法是给
  `/etc/udev/rules.d/forklift_camera.rules` 补 `ATTR{index}=="0"`。

完整调查记录见 `results/camera_intrinsics_in_project.md`。

## 相机外参（provisional）

外参已经有了**临时估算值**并写进 MuJoCo，但精度还不够做精细抓取，后续会重做。
本节说明这份值从哪来、怎么用、误差多大、以及重做时该避开哪些坑。

### 现状

| 项 | 值 |
| --- | --- |
| 相机位置（基座系） | `(0.335999, 0.051555, 0.362488)` m |
| 俯角 / yaw | 79.39° / −96.07° |
| 到基座距离 | 0.340 m |
| 重投影残差 mean/p95/max | 1.37 / 2.64 / 3.41 px |
| **实际不确定度** | **±14.5 mm（411 mm 上 3.5%）** |

残差小不代表准。那三个像素数只是「同一张图内点与点的自洽性」，不含桌面测量误差。
真实误差约 ±14.5 mm，主要来自：卷尺读数与直角摆放（±5 mm）、25 mm 方格尺寸本身的
标定（±13.5 mm，格距与总跨距互相矛盾 6.6%）。30 mm 的立方体配上 ±14.5 mm，
余量约半个物体，所以这份外参只适合可视化、不适合精细抓取。

数据与推导过程在 `results/extrinsics_hf868.json`。

### 在 MuJoCo 里怎么用

外参写在 **`models/so101/scene_hf868.xml`**，它是一个**叠加场景**：`include`
原 `scene_fixed.xml`，只在 `worldbody` 里追加一台 `hf868_camera`。

```python
import mujoco

m = mujoco.MjModel.from_xml_path("models/so101/scene_hf868.xml")
# 渲染 640x480（默认离屏帧缓冲上限就是 640x480）
```

**为什么不直接改 `scene_fixed.xml`**：那个文件的 SHA256 被
`experiments/*/source_hashes.json`、`protected_hashes.json` 锁定，还被
`datasets/raw_v3` 下每个 `episode.json` 记录。改一个字节就会让 `validate.py`
的断言失败、历史数据集无法复现。所以保持原文件零改动。

相机参数与实机的对应关系：

| 实机 | MuJoCo | 说明 |
| --- | --- | --- |
| `fx, fy = 326.68129, 326.27539` | `focalpixel` | 实测与渲染差 0.4 px，精确生效 |
| `cx, cy = 319.20034, 248.66437` | **无对应** | MuJoCo 不支持主点偏移，见下 |
| 640×480 | `sensorsize` + `resolution` | `sensorsize` 只定 4:3 宽高比 |
| 位置 + 姿态 | `pos` + `xyaxes` | `xyaxes` 由 OpenCV 的 R 换算：x 轴照抄，y 轴取反（y 下→y 上），z 轴由叉乘得 |

### MuJoCo 的两条能力边界（实测 MuJoCo 3.15.0）

**1. 主点偏移写不进去。** 设 `principal` 会让渲染直接输出全空图（即使取
`0.5 0.5`）；不设时 MuJoCo 一律假定主点在画面正中。HF868 的实机主点是
`(319.20, 248.66)`，正中是 `(320, 240)`，即偏移 `(−0.80, +8.66)` px。

所以渲染结果相对实机会整体偏移这么多像素。**实机图 = MuJoCo 图 + (dx, dy)**，
换算函数是 `scripts/hf868_sim_camera.py` 里的 `hf868_principal_shift()`。
这个 8.66 px 在立方体深度（353 mm）上换来约 9.4 mm 的等效位置偏差，**小于上面
±14.5 mm 的外参不确定度，暂不补偿**；等外参更准后再做平移。

**2. 改渲染分辨率不改变 FOV，但会改变等效像素焦距。** `focalpixel` 是「给定
`resolution` 下的像素焦距」，归一化坐标验证过 FOV 恒定。所以渲染 320×240 时
等效 `fx` 减半为 163.34，要和实机比较时必须换算。另外默认离屏帧缓冲上限
640×480，要更大得在 `<visual><global offwidth/offheight>` 抬上限。

### 重做时要知道的四件事

**（一）这块棋盘格是 7×10 内角点、25 mm 方格**，但打印比例可疑。
`references` 里 `checkerboard.pdf` 的页面尺寸是 2520×3168 pt = **889×1118 mm**，
不是 A4，说明打印时被缩放过。重做时**先用卡尺量一个格子的实际边长**，
不要假设 25 mm。当时卷尺测量内部就有一个 6.6% 的矛盾
（`213.2/9 = 23.68 mm/格` vs `152.0/6 = 25.33 mm/格`），可能就来自这里。

**（二）单张平面棋盘格有不可消除的镜像/转置二义性**，两种解的复投影残差
完全相同。当时用「帧差法」判别的：做一次机械臂微小运动，比较两个候选解
预测的腕部像素位移与实测位移 —— `沿Y镜像` 预测 (384,501) 对观测 (419,451)
差 61 px，`沿X镜像` 差 494 px，据此排除。以后凡是单图 planar PnP，都要
独立验证这一步，不能只看残差。

**（三）ArUco 手眼标定这条路当时没走通，原因是几何而非参数。** 相机装在
桌沿俯视桌面，在 `z=0` 上只能覆盖 `x ∈ [−34, 610] mm`、`y ∈ [−500, 319] mm`，
而基座原点投影到 `(345, 536)`、折起的手臂投影到 `(404, 580)` —— 都在 480 行
**画面之外**。所以手臂折叠时腕部靶标根本拍不到，必须先伸到桌面上方
（约 `x≈0.30–0.35 m, z≈0.15–0.20 m`，投影落在 290–210 行才是画面内）。

已经验证过的 ArUco 结论，重做时可直接用：

- 50 mm 靶标在 320 mm 处约 50 px；**倾斜 0°–70° 都能检出，75° 起失败**；
- 但**正对相机（<10°）时两解残差几乎相同，位姿会跳**（easy_handeye 的
  troubleshooting 也点名了这个失败模式）。**最佳区间 15°–40°**，
  和平贴在腕部顶面天然形成的 15.6° 夹角正好吻合；
- 靶标贴附位置在数学上无所谓（`AX=XB` 会把它当副产品一起解掉），
  **唯一硬要求是刚性**：不能翘边、不能晃。

**（四）走位务必先做可达性检查。** 当时下发了一个实际不可达的目标
（`elbow_flex` 要到 −36°，实测卡在 36.7°），脚本又写了「误差不收敛就继续发」
的循环，硬顶 60 次把供电拉到欠压（`Input voltage error`），属于堵转。
重做时的规则：离线先算可达性、误差连续 3 步不降就立即中止并回退、
`max_relative_target ≤ 3°`、大幅改姿态前先问人。更省事的办法是**人手动摆位、
程序只读相机和关节角，完全不给力矩**。

### 重做后的收尾

拿到新的外参后，只需要改 `scene_hf868.xml` 里 `hf868_camera` 的三个属性
`pos` / `xyaxes` / `focalpixel`，更新本节表格和 `results/extrinsics_hf868.json`。
`scene_fixed.xml` 始终不动。

## 使用环境

```bash
source .venv/bin/activate
python -m so_ws.scripts.verify_model
```

在 macOS 上打开交互式 Viewer 时，使用工作空间脚本和 `mjpython`：

```bash
./.venv/bin/mjpython scripts/view_so101.py
```

该命令打开第一版固定抓取场景 `models/so101/scene_fixed.xml`。

Viewer 启动时固定加载 XML 中的 `folded_ready` keyframe；不要使用直接加载 XML 的通用
Viewer 命令，否则 MuJoCo 不会自动选择 keyframe。

## 抓取、盒内放置和返回初始姿态

在工作空间根目录执行：

```bash
./.venv/bin/mjpython scripts/launch_pick_place.py
```

速度配置：

```bash
# 快速仿真调试
./.venv/bin/mjpython scripts/launch_pick_place.py --profile simulation_fast

# 真机准备阶段的保守仿真
./.venv/bin/mjpython scripts/launch_pick_place.py --profile real_safe
```

两个配置的控制时序如下；`real_safe` 的速度上限是用于真机准备阶段的保守仿真值，
不是已经测量过的 SO-101 电机极限，接真机前仍需根据实际舵机反馈和电流重新标定。

| 配置 | 物理步/控制 tick | 控制频率 | 单段 Cartesian tick | 关节目标变化上限 |
|---|---:|---:|---:|---:|
| `simulation_fast` | 8 | 60 Hz | 36 | 8.0 rad/s |
| `real_safe` | 16 | 30 Hz | 64 | 1.5 rad/s |

设计依据是 LeRobot 默认 30 FPS 的策略/记录循环，以及通过插值提高机器人命令频率的
机制：[rollout 配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/configs.py)
和[推理文档](https://github.com/huggingface/lerobot/blob/main/docs/source/inference.mdx)。

窗口先展示 `folded_ready` 初始姿态，然后依次打开夹爪、接近方块、下降、闭合、抬升、
移动到盒子上方、下降到盒内、释放方块，最后回到 `folded_ready`。演示结束后窗口停留在
最终姿态，关闭窗口即可退出。
方块边长为 30 mm，质量为 20 g；不指定任务时使用场景 keyframe 里的固定初始位置
`(0.25, 0, 0.015)` m，采集时由 `src/task_sampling.py` 采样初始状态（见下文）。
成功条件为：保持后方块抬升至少 70 mm、接触两侧夹爪、且不接触桌面；释放后方块中心
位于盒内；机械臂回到 `folded_ready` 附近（关节误差不超过 0.05 rad，夹爪误差不超过
0.03 rad）。回程阶段夹爪同步闭合到初始控制值，掌部碰撞体不参与碰撞，避免回零后再
触发一次 keyframe 修正；脚本会把实际误差打印出来。

可以用 `--seed` 让脚本按采样器随机一个任务，或直接指定任务：

```bash
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --seed 20261010
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --cube-xy 0.22 0.05 --cube-yaw-deg 30
```

`run()` 返回结构化结果 `RunResult`（`success/grasp_ok/placed/returned/failure_phase/
failure_reason`），采集器据此区分「抓取失败」「放置失败」「回程失败」和 IK 中断。

无窗口验证命令：

```bash
./.venv/bin/python scripts/launch_pick_place.py --no-viewer
```

抓取控制点、动作阶段和夹爪接触垫参数参考
[ICL-so101 抓取检查脚本](https://github.com/desmondzee/ICL-so101/blob/main/sim/check_grasp.py)
与其[夹爪碰撞模型](https://github.com/desmondzee/ICL-so101/blob/main/sim/so101/so101_new_calib_camera.xml)。
当前 `target_box` 已启用底板和四面墙碰撞。脚本从盒口上方释放方块，让它自然落到底板，
并检查方块中心是否位于盒内范围。抓取动作采用正方体四个等效 yaw 候选、minimum-jerk
位移和闭合后微抬升验证；候选选择与动作接口位于
`src/controllers/square_grasp_planner.py`。

本地适配参考桌面上的 `lx_arm101` 规划器和 `pick_primitive.py`，只采用抓取、碰撞和轨迹
平滑相关逻辑，不引入其中的地图建造、工作空间扫描、训练或数据集批处理脚本。

动作速度按 profile 设置；两个 profile 都在 MuJoCo 物理步内继续插值，避免 position
actuator 接收到阶跃 setpoint；夹爪采用
minimum-jerk 闭合并根据双侧接触提前结束。这样可以缩短动作时间，同时保留接触稳定性。
这个设计与 [pick-101 的渐进闭合和接触检测](https://github.com/temperancee/so-101_mujoco/blob/main/README.md)、
[MuJoCo 的 setpoint slew-rate 说明](https://mujoco.readthedocs.io/en/3.7.0/_static/dcmotor.pdf)
以及 [SO-101 quintic trajectory 项目](https://github.com/yayabash/so101-classical-robotics)
中的连续轨迹思路一致。

## 本地数据集采集

本机只负责生成原始专家轨迹，运行：

```bash
./.venv/bin/python scripts/collect_pick_place_dataset.py \
  --episodes 20 \
  --output-dir datasets/raw_v1
```

`--output-dir` 没有合理的默认值可依赖：采集器的默认值仍是 `datasets/raw`，而第一版正式
数据约定写在 **`datasets/raw_v1`**，所以命令里显式带上它。`datasets/raw` 现在只保留格式
说明 `README.md` 和占位文件，里面已经没有数据了。

采集前会检查 `git status --porcelain`，工作区不干净时拒绝采集；调试时用 `--allow-dirty`
跳过，但 `episode.json` 里会记 `git_dirty: true`。不指定 `--start-index` 时自动从已有最大
编号 +1 开始，种子由编号推导，所以中途 Ctrl-C 之后直接用同样的命令续采即可，不会重复种子、
也不会留下残缺目录（`.tmp_episode_*` 会在下次启动时清理）。

每个 episode 保存为一个目录，包含：

```text
episode_000000/
  episode.json
  state_action.npz
  front_camera/*.png
  wrist_camera/*.png
```

成功的 episode 放在 `<output-dir>/episode_*`，失败的放在 `<output-dir>/failed/episode_*`，
失败轨迹的数据同样完整，可以留作负样本；两者的编号都会写进 `<output-dir>/index.jsonl`，
每批采集再写一份 `<output-dir>/manifest.json`。采集默认 320×240（`--width/--height` 可改），
`front_camera` 和 `wrist_camera` 两路同分辨率。格式细节见
[`datasets/raw/README.md`](../datasets/raw/README.md)（该文件是各批次共用的格式说明）。

`state_action.npz` 包含时间戳、6 个关节的 `qpos/qvel`、`action`、方块位姿和动作阶段。
`action` 是限速后的绝对关节位置目标，在下一个控制 tick 内线性插值执行；观测在该 tick 的
物理步之前记录，因此 `obs[t]` 与 `action[t]` 严格同拍（`obs_action_alignment` 字段）。
采样频率为 30 Hz（`real_safe`，`control_period_sec` 由 `physics_per_tick × timestep` 算出）。
开环回放测试 `tests/test_action_alignment.py` 要求回放轨迹与记录误差小于 1e-9，用来防止
观测/动作再次错拍。

格式版本为 `local_raw_v3`。三个版本的动作语义相同（绝对关节位置目标、`obs[t]` 与
`action[t]` 同拍），差异在采样率上，因此不能混用：

| 版本 | 控制频率 | 默认分辨率 | 说明 |
|---|---:|---|---|
| `local_raw_v1` | 31.25 Hz | 160×120 | 动作错了一拍，已废弃 |
| `local_raw_v2` | 31.25 Hz | 160×120 | 修正了 obs/action 对齐 |
| `local_raw_v3` | 30 Hz | 320×240 | `timestep` 改为 1/480 s，`real_safe` 控制周期恰好 1/30 s |

控制频率从 31.25 Hz 改成整 30 Hz 的原因是与 LeRobot 默认 30 FPS 的策略/记录循环对齐。
`scene_fixed.xml` 的 `timestep` 取 `0.0020833333333333333`，使 `16 × timestep` 精确等于
双精度的 `1/30`，`manifest.json` 里的 `sampling_hz` 是精确的 `30.0` 而不是 `29.999…`
（`tests/test_control_rate.py` 断言这一点）。各 profile 中以物理步计的相位时长按原来的
秒数换算后四舍五入到 `physics_per_tick` 的整数倍，见 `src/controllers/motion_profiles.py`
里的逐项注释。

### 图像分辨率与相机取景

采集默认分辨率是 320×240。取景用下面的脚本检查，它会渲染采样范围四个角和中心各一张
front 和 wrist 图，并测量方块在 front 图中的像素尺寸：

```bash
./.venv/bin/python scripts/check_camera_framing.py
```

测量方法是把方块 geom 的 alpha 临时置 0 再渲染同一帧，两次渲染的差异就是方块的可见轮廓。
这样不需要颜色阈值（夹爪的红色和方块很接近），也不依赖相机内参约定。**测量时必须临时
关掉 `mjRND_REFLECTION`**：桌面材质有 `reflectance=0.15`，倒影会和方块一起参与像素差异，
实测把轮廓高度从 19 px 撑到 29 px、面积虚高 56%。关阴影没有影响。

320×240 下 front 图的实测结果（`fovy=35°`）：

| 位置 | 相机距离 | 轮廓宽×高 | 面积 | 面积等效边长 | 针孔单边参考 | wrist 能看到方块 |
|---|---:|---|---:|---:|---:|---|
| r=0.20, θ=-30° | 0.710 m | 20×23 px | 426 px | 20.6 px | 16.1 px | 是 |
| r=0.20, θ=+30° | 0.710 m | 20×23 px | 426 px | 20.6 px | 16.1 px | 否 |
| r=0.24, θ=-30° | 0.679 m | 21×25 px | 489 px | 22.1 px | 16.8 px | 否 |
| r=0.24, θ=+30° | 0.679 m | 21×25 px | 488 px | 22.1 px | 16.8 px | 否 |
| r=0.22, θ=0°（中心） | 0.669 m | 18×25 px | 450 px | 21.2 px | 17.1 px | 是 |

轮廓高比宽大是因为俯视角度下能看到方块顶面；面积等效边长偏大是因为轮廓不是正方形。
针孔参考 `f = (H/2)/tan(fovy/2) = 289.7 px` 用的是实测校准过的 f：把方块沿世界 y 平移
20 mm，质心像素位移反推 `f ≈ 297.7 px`（差 2.8%），而按水平视场角算会得到 386.3 px，
明显不符，所以 `cam_fovy` 是竖直视场角。

**投影计算里 `cam_xmat` 的约定容易搞错**：`np.asarray(data.cam_xmat[id]).reshape(3, 3)`
之后，**列**才是相机的 x/y/z 轴在世界系下的方向（行是转置）。用错约定会把方块算到画面外
几百像素的地方；验证方法是拿已知点的投影去对实测像素 —— 例如 `(0.20, -0.10, 0.015)`
在 `fovy=45°` 下投影到 u≈117.8、v≈180.4，与实测的方块轮廓 u≈110–125、v≈171–199 吻合。

#### 取景收紧：fovy 45° → 35°

第一版按 `fovy=45°` 采过一批数据，最保守读数（针孔单边 12.2 px）只比 12 px 阈值高 0.2 px，
而画面里大片留白。确认后把 `front_camera` 的 `fovy` 从 45° 收到 35°，`pos` 与 `xyaxes`
不动：

| | fovy 45°（旧） | fovy 35°（当前） |
|---|---:|---:|
| 方块轮廓宽×高 | 16×19 px | 20×23 px |
| 方块单边（针孔最保守读数） | 12.2 px | 16.1 px |
| 距 12 px 阈值的余量 | 0.2 px | 4.1 px（相对余量 34%） |
| 手臂在画面内 | 完整（上 58 / 下 88 px） | 完整（上 39 / 下 13 / 右 16 px） |

收紧的上限由「内容必须留在画面内」决定。把整个演示 577 帧里手臂所有 body 的投影统计一遍，
再继续收紧就会裁掉手臂：

| 配置 | 方块 | 手臂上边距 | 结果 |
|---|---:|---:|---|
| fovy 45° | 17.4 px | 58 px | 完整 |
| **fovy 35°** | **22.9 px** | **39 px** | **完整（当前）** |
| 再加 tilt 31°、fovy 30° | 26.3 px | -13 px | 手臂顶部裁掉 |
| tilt 35°、fovy 28° | 28.0 px | -60 px | 手臂顶部裁掉 |

搜索过「内容四边留白 ≥ 10 px 前提下方块最大」的最优解，是 tilt 27° + fovy 34°，方块
23.5 px —— 只比只改 fovy 多 0.6 px，不值得多改一个参数，所以没有采用。

相机不参与物理：改 `fovy` 只影响渲染，动力学、专家成功率和 `obs/action` 对齐都不受影响，
5×5×4 标定无需重跑。变化的是 `scene_sha256`，因此不同取景的原始数据在元数据里天然可区分。

wrist 相机在 `folded_ready` 初始姿态下只在部分位置能看到方块，这是正常的：它服务于
末端逼近阶段，不是用来在开局观测方块的。

单条 episode 的实测开销（320×240，`real_safe`，577 帧）：

| 分辨率 | 帧数 | 单条总大小 | 其中 npz | PNG 每帧 | 单条耗时 |
|---|---:|---:|---:|---:|---:|
| 160×120 | 581 | 10.07 MB | 92 KB | 17.3 KB | 12.11 s |
| 320×240 | 577 | 23.37 MB | 92 KB | 40.7 KB | 13.06 s |

320×240 的像素数是 160×120 的 4 倍，磁盘占用涨到 2.3 倍（PNG 压缩吸收了余量），
耗时只涨 7.8%。帧数从 581 降到 577 有明确原因：`lift_hold` 少 1 tick、`release_settle`
少 1 tick（设计值），夹爪闭合阶段因物理步变细、双侧接触提前判定而少 2 帧。

### 任务初始状态采样与范围标定

初始状态由 `src/task_sampling.py` 用极坐标采样：`r ∈ (0.20, 0.24) m`、
`θ ∈ (-30°, 30°)`、方块 yaw ∈ `[0°, 90°)`（正方体 90° 对称），盒子默认不动。
`TaskSample` 是纯数据，写进 `episode.json` 后可以完整复现同一个初始状态。

执行之前先做可达性预检：从复位状态出发依次预解
`pregrasp → grasp → lift → above_target → place`，每段用上一段的结果作为初值。
抓取点、预抓取点和释放点用 5 mm 硬标准（实测都在 0.1 mm 以内）；「抬升」和
「移动到盒子上方」是让开障碍的中间路点，目标点位于 5 自由度手臂在「末端朝下」约束下的
灵巧工作空间之外，实测最优误差 6.9–20 mm（300 次随机初值重启也找不到更优解，
`wrist_flex` 已顶到关节限位），因此按 25 mm 判定，仍低于 `move_to` 的 30 mm 失败阈值。
此外还会检查复位后方块是否与折叠的夹爪穿模：`folded_ready` 的夹爪停在桌面
x≈0.13–0.19 m、y≈0 处，方块采样到那里会在物理第一步被弹开（实测穿透 4–20 mm、
方块被推出 5 cm），这种样本直接丢弃并换下一个子种子重采样。

范围用网格重新标定过：

```bash
./.venv/bin/python scripts/calibrate_workspace.py \
  --output results/workspace_calibration_full.json
```

在 r × θ × yaw 上取 5×5×4 网格（r = 0.16/0.18/0.20/0.22/0.24 m，
θ = -30/-15/0/15/30°，yaw = 0/22.5/45/67.5°）跑完整专家演示，结果：

| 半径 | 执行次数 | 成功 | 执行前丢弃 | 执行后失败 |
|---|---:|---:|---:|---:|
| 0.16 m | 12 | 9 | 8 | 3 |
| 0.18 m | 12 | 11 | 8 | 1 |
| 0.20 m | 14 | 14 | 6 | 0 |
| 0.22 m | 20 | 20 | 0 | 0 |
| 0.24 m | 20 | 20 | 0 | 0 |

上表是把控制频率改成整 30 Hz（`timestep` 1/480 s）之后重跑的结果。逐格对比改频率之前的
31.25 Hz 基线，100 个格子里只有 1 个不同：`r=0.16 m, θ=+30°, yaw=45°` 仍然失败，但失败
阶段从「抬升」变成「下降」；其余 99 格的 `success` 与 `discarded` 完全相同，各半径的
执行/成功/丢弃计数也就没有变化。改频率前耗时 94.6 s，改后 90.2 s。

最初按探测结果取 `r ∈ (0.16, 0.24)`，100 个格子成功 74 个，可达样本内成功率 94.9%，
达不到 95%；4 次真正的专家失败全部落在 r ≤ 0.18 m 上（30 Hz 下是 3 次「抬升没夹住方块」
加 1 次「下降到盒内」）。缩小到 `r ∈ (0.20, 0.24)` 后 60 个格子执行 54 次全部成功（100%），
6 次被预检丢弃（10.0%，低于 30% 上限），因此默认范围定为 `(0.20, 0.24)`。
标定脚本的参数可以自定义，报告写到 `results/`（该目录不进 Git）。

### 俯视抓取的 yaw 对齐

`folded_ready` 下来自 `wrist_roll` 的姿态与位置/朝下约束耦合，不加约束时夹爪闭合方向
不随方块 yaw 变化（实测固定停在 -1.7°，最大偏差 43°）。`solve_ik` 因此增加了 `target_yaw`：
在位置和朝下约束之外加一行世界 z 轴角速度约束来驱动夹爪闭合方向（实测闭合方向是 gripper
局部的 x 轴，由 `moving_jaw_tip` 的开合位移确定）。关节角与世界 yaw 的偏移随姿态变化
（yaw=0 时 +2.7°，yaw=75° 时 +5.4°），所以对齐必须在世界坐标系里做，不能写成固定偏移常量。
「接近」「下降」「抬升」三段锁定 `target_yaw`，放置和回程不约束。验收：固定 (0.25, 0) 下
方块 yaw 取 0/15/30/45/60/75° 全部成功，yaw 误差小于 0.01 rad（见
`tests/test_yaw_alignment.py`）。

标定网格的 yaw 只取到 67.5°，而采样范围是 `[0°, 90°)`。正方体在 90° 处与 0° 几何等价，
但 yaw 对齐误差随姿态变化，所以 67.5°–90° 这一段单独补测：在采样范围 `r = 0.20/0.24 m`、
`θ = ±30°` 的四个角上取 yaw = 80° 和 89°，8 个组合的完整演示全部成功；同一批 yaw 下
方块上方 12 mm 处的 IK yaw 误差也小于 0.01 rad。

## LeRobot 格式转换

采集和转换是分开的，原有转换与训练仍可在服务器上做。本机现已使用
`third_party/lerobot_v061/` 中的 LeRobot 0.6.1；转换脚本仍保持延迟导入，
只有真正写数据集时才加载 LeRobot，`--dry-run` 仍不需要它。

```bash
# 只扫描和校验，不需要 LeRobot
./.venv/bin/python scripts/convert_to_lerobot.py --dry-run --raw-dir datasets/raw_v1_pilot

# 真正转换并读回校验（需要 LeRobot，在服务器上执行）
./.venv/bin/python scripts/convert_to_lerobot.py \
  --raw-dir datasets/raw_v1_pilot \
  --repo-id local/so101_pickplace_v1 \
  --root datasets/processed/so101_pickplace_v1 \
  --verify
```

转换只处理 `<raw-dir>/episode_*`（跳过 `failed/`），并要求每条 episode 同时满足：
`format == local_raw_v3`、`git_dirty == false`、`success == true`、
`image_size == [320, 240]`、`sampling_hz == 30.0`。不满足的会被逐条列出原因后跳过，
不会静默混进来 —— 这一步正好挡住 `datasets/raw_v1` 里可能混进的旧格式数据。

`--max-episodes N` 只取**编号最小的 N 条成功 episode**，用来生成 50 / 100 / 200 的嵌套
子集做数据量消融：

```bash
for N in 50 100 200; do
  ./.venv/bin/python scripts/convert_to_lerobot.py \
    --raw-dir datasets/raw_v1 --max-episodes "$N" \
    --repo-id "local/so101_pickplace_v1_$N" \
    --root "datasets/processed/so101_pickplace_v1_$N" --verify
done
```

因为取的是升序前缀，三个子集天然嵌套（50 ⊂ 100 ⊂ 200）。被格式校验拒掉的 episode
不占名额（N 是对成功 episode 而言的）。`--verify` 会用与 `convert` 完全相同的子集去比对，
不会拿错 episode。不加这个参数就是全部成功 episode。

输出是 LeRobot v3 数据集：`fps=30`、`robot_type="so101_follower"`、唯一任务文本
`"Pick up the cube and place it in the box"`。特征定义：

| 特征 | dtype | shape | names |
|---|---|---|---|
| `observation.state` | `float32` | `(6,)` | `shoulder_pan.pos` … `gripper.pos` |
| `action` | `float32` | `(6,)` | 同上 |
| `observation.images.front` | `video` | `(240, 320, 3)` | `height`/`width`/`channels` |
| `observation.images.wrist` | `video` | `(240, 320, 3)` | 同上 |

特征的 `info` 子块（`video.codec` / `video.pix_fmt` 等）不用自己写：LeRobot 在编完第一条
episode 的视频后会反查实际码流，由 `update_video_info` 写回 `meta/info.json`。
`add_frame` 的 `task` 键是必需的；`timestamp` / `frame_index` / `episode_index` 由 LeRobot
自己填，不能出现在传入的字典里。

### 单位换算与夹爪刻度

5 个手臂关节由弧度换算成度（与 LeRobot SO-101 默认的 `use_degrees=True`、
`MotorNormMode.DEGREES` 一致）。

本节描述数据转换与 ACT 使用的夹爪刻度。官方手柄处理链按自己的开合方向做端点桥接，
两种用途的映射不要混用，见 [手柄适配说明](gamepad_mujoco.md#必需的设备与仿真桥接)。

夹爪按仿真关节限位线性映射到 0–100：

```text
夹爪刻度 = (q - low) / (high - low) * 100
low, high = 场景里 gripper 关节的限位 = (-0.1745, 1.7453) rad
```

限位从 `scene_fixed.xml` 读，不写死常量，否则改了模型会导致整个夹爪刻度偏移。实测取值：

| 量 | 仿真关节值 | 映射后 |
|---|---|---|
| `GRIPPER_CLOSED`（闭合指令） | -0.17 rad | 0.23 |
| `GRIPPER_OPEN`（张开指令） | 1.20 rad | 71.60 |
| 实际 `action[5]` 区间 | [-0.17, 1.20] | [0.23, 71.60] |
| 实际 `qpos[5]` 区间 | [-0.0001, 1.261] | [9.09, 74.77] |

注意仿真从不把夹爪开到 100：`GRIPPER_OPEN = 1.20` 低于关节上限 1.7453；而夹住方块时
`qpos[5]` 停在约 9 而不是 0，因为闭合指令 -0.17 本身就略高于关节下限。

> **这是仿真近似，接真机前需要按真机标定复核。**

「0 为闭合」这个方向**没有**在 LeRobot 源码里得到确认，实际情况是源码根本不固定这个方向，
所以必须在真机上复核：

- `so_follower.py` 给夹爪用的是
  [`MotorNormMode.RANGE_0_100`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)；
- `motors_bus.py` 的 `_normalize` 是
  [`((bounded_val - min_) / (max_ - min_)) * 100`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py)，
  即 0 对应标定记录的 `range_min`、100 对应 `range_max`，`drive_mode=1` 时整体翻转；
- 而 `range_min` / `range_max` 是标定时用户手动把关节扫过全行程记录下来的**原始编码器读数**
  （`record_ranges_of_motion`），哪一端是闭合取决于装配方向和 `homing_offset`。

支持「0 为闭合」的旁证：`isaka1022/so101-sim2real` 把自己环境的 `grasp` 定义成
`0` 全闭、`1` 全开（见其 Action space 文档）；官方
[`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
的夹爪只用到 0–33，静止在 1.5 附近、抓取时抬到 20–27，与该方向一致。但这些都不是 LeRobot
的规范性定义，所以接真机时要用真机标定文件里的 `range_min`/`range_max` 和实测的开合方向
重新核对一次刻度，必要时把映射翻转。

### 时间戳与读回校验

时间戳写 `frame_index / 30`，并断言与原始 `timestamp` 一致（容差 1e-9 s）。MuJoCo 的
`data.time` 是逐物理步累加的，所以只能按容差比：实测 577 帧上累计漂移 1.8e-12 s，
远小于容差，但错一拍（0.033 s）必然会失败。

`--verify` 用 `LeRobotDataset` 读回，对每条 episode 抽查第 0 帧和最后一帧，把
`observation.state` / `action` 与原始数据换算后按 1e-4 容差比对，并检查图像形状。
v3 的 episode 边界在 `meta.episodes` 的 `dataset_from_index`（v2 的 `episode_data_index`
已经不存在）。

上述“扫描/校验/换算测试 16 条通过”是早期环境的记录，当时本机尚未集成 LeRobot。
当前本机已接入 LeRobot 0.6.1；本次整理时 `datasets/processed/` 仅有占位文件，
没有据此确认任何本机转换结果。真正的写入和读回仍需对具体批次执行 `--verify`，
不能用历史扫描测试代替。

## 第一版数据集 v1 的试点批次

试点批次采在 `datasets/raw_v1_pilot`（不写进 `datasets/raw/`），工作区提交干净后采集，
因此全部 50 条的 `git_dirty` 都是 `false`、`git_commit` 是 `c26789d3ba96`（取景收紧到
`fovy=35°` 之后的那次提交），`scene_sha256` 以 `a4a5a282f2a52a06` 开头。

```bash
./.venv/bin/python scripts/collect_pick_place_dataset.py \
  --episodes 50 \
  --output-dir datasets/raw_v1_pilot
```

| 指标 | 结果 |
|---|---:|
| episode 数 | 50（成功 50，失败 0） |
| 成功率 | 100.0% |
| 预检丢弃样本 | 2 个（采样尝试 52 次，丢弃率 3.8%） |
| 整编号跳过 | 0 |
| 失败阶段 | 无 |
| 每条帧数 | 全部 577 帧（合计 28850 帧，按 30 fps 约 16.0 分钟） |
| 总磁盘占用 | 1.28 GB（平均 25.67 MB/条） |
| 总耗时 | 737.1 s（12.3 分钟，平均 14.74 s/条） |

每条的数据完整性一致：`format=local_raw_v3`、`sampling_hz=30.0`、`image_size=[320,240]`、
`git_dirty=false`。

采样覆盖（r、θ、yaw 全部落在标定区间内）：

| 量 | 实测范围 | 区间跨度占比 | 卡方均匀性（6 桶） |
|---|---|---:|---|
| r | 0.2002–0.2387 m | 96.2% | χ²=3.44, p=0.842 |
| θ | -27.44°–+27.44° | 91.5% | χ²=2.08, p=0.838 |
| yaw | 1.23°–89.99° | 98.6% | χ²=12.88, **p=0.025** |

覆盖分布与取景收紧前那一批**完全相同** —— 种子由编号推导，同一编号永远对应同一个任务，
所以换相机不影响任务采样，这也顺带验证了采集流程的可复现性。

yaw 这一项在 p=0.025 上偏离均匀（`[0°,15°)` 落进 17 条即 34%，均匀预期 16.7%）。这**不是
采样器的偏差**：直接用同一套 `derive_seed`/`sample_task` 在无物理的情况下抽样，n=3000 时
χ²=0.50、p=0.992，n=30000 时 p=0.549，说明采样器本身均匀；而且同时检验 3 个分布时，
Bonferroni 校正后的阈值是 0.0167，p=0.025 也过不了。结论是 50 条样本的偶然涨落，
正式批次（300 条）不需要为此改采样策略。

按试点实测线性外推，正式批次 300 条约需 **7.7 GB 磁盘、74 分钟**。

**这份历史试点记录没有完成本机转换**：当时 50 条通过扫描校验（接受 50、拒绝 0），
没有执行真正写入和 `--verify` 读回。当前本机已有 LeRobot，下面保留的是针对旧试点目录的
历史转换命令；当前数据的转换入口见 [项目首页](../README.md#采集与推理)。

```bash
./.venv/bin/python scripts/convert_to_lerobot.py \
  --raw-dir datasets/raw_v1_pilot \
  --repo-id local/so101_pickplace_v1 \
  --root datasets/processed/so101_pickplace_v1 \
  --verify
```

## 分层采样

第一版数据集用分层采样而不是完全均匀随机：r、θ、yaw 各切 4 档，共 64 格，格子按
`编号 % 64` 轮转分配，格内均匀抖动。这样任意前 N 条里每格的条数最多差 1，前 64 条每格
恰好 1 条 —— 300 条时每格 4–5 条，不会出现某个角被抽空的情况。

```bash
./.venv/bin/python scripts/collect_pick_place_dataset.py \
  --episodes 300 \
  --output-dir datasets/raw_v1          # 默认 --sampler-mode stratified --bins 4 4 4
```

- **种子与复现**：种子仍由 `derive_seed(base_seed, 编号, attempt)` 推导，同一编号永远对应
  同一个任务，续采不会重复。
- **丢弃重采不换格**：格子只由编号算一次，重采只换派生种子，所以不可达样本在同一格里
  重采，不会漂到别的格子。
- **记录**：`episode.json` 里有顶层 `sampler_mode`，以及 `sampler.mode`、`sampler.bins`
  和 `sampler.sample.cell`。`manifest.json` 也带 `sampler_mode`。
- 需要回到完全均匀随机时用 `--sampler-mode uniform`（此时 `cell` 为 `null`）。

`SamplerConfig.mode` 本身默认是 `uniform`，只有采集器默认开 `stratified`：这样
`sample_task(seed)` 这种不指定格子的既有调用（`pick_cube.py` 的单任务模式、单测）不受影响。

## 评估布局（eval_layouts_v1）

评估用的 50 个布局由独立脚本生成，只做采样与可达性预检，**不采集**：

```bash
./.venv/bin/python scripts/generate_eval_layouts.py
```

产物 `datasets/evaluation/eval_layouts_v1.json`（不进 Git）：

| 项 | 值 |
|---|---|
| 布局数 | 50 |
| 覆盖格子 | 50 / 64 |
| 未覆盖格子 | 4, 9, 13, 18, 22, 27, 31, 36, 41, 45, 50, 54, 59, 63 |
| 可达性预检 | 全部通过，最差误差 20.90 mm |
| 需要重采的布局 | 1 个（只在同一格内重采） |
| `eval_base_seed` / `train_base_seed` | 20261011 / 20261010 |
| sha256 | `58d3297fedce6cf8023df65f061861927dae5f39409ca91a83051245d03bad1f` |

- **种子空间分离**：脚本会断言评估用到的每一个派生种子都不出现在训练前 10000 个编号
  （× 16 次尝试）的种子集合里；实测交集为 0。训练集合是 159997 而不是 160000，是因为按
  生日问题预期 160000 次派生里约有 3 次碰撞、去重后少了 3 个。
- **格子摊开**：评估的格子用 `(i × 64) // 50` 在整个 64 格里均匀摊开，而不是像训练那样
  从 0 开始连续取 50 个 —— 否则未覆盖的 14 格会集中在一角。50 < 64 所以必然有 14 格
  没有评估布局；要全覆盖就把 `--count` 改成 64。
- **确定性**：文件内容只由种子、配置和可达性决定，不含时间戳或 commit 号，重复运行得到
  逐字节相同的文件。所以上面记的 sha256 可以用
  `shasum -a 256 datasets/evaluation/eval_layouts_v1.json` 复验。
