# lx-so101

SO-101 机械臂的 MuJoCo 开发工作空间，包含 F310 手柄遥操作、官方 LeRobot IK 接入、
专家抓放、轨迹采集与 ACT 推理入口。说明、配置注释和开发记录使用中文。

这是可继续开发的源码版本：保留机器人 MJCF/URDF/STL、配置、测试、相机标定示例和文档，
不包含采集轨迹、训练权重、实验环境或原工作区的 Git 历史。

导航：[安装](#安装) · [启动](#启动) · [手柄与-ik](#手柄与-ik) ·
[采集与推理](#采集与推理) · [验证](#验证) · [目录与资料](#目录与资料)

## 安装

本次验证环境为 macOS Apple Silicon、Python 3.12；Mac 窗口入口使用 `mjpython`。
其他系统的窗口和手柄兼容性尚未由本次交付验证。
先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后：

```bash
git clone https://github.com/lixi181/lx-so101.git
cd lx-so101
uv sync --frozen --python 3.12 --extra gamepad --extra dev
```

后续命令均在仓库根目录执行。依赖按 [uv.lock](uv.lock) 固定；
LeRobot 0.6.1 源码已包含在 `third_party/lerobot_v061/`，安装不依赖原机器路径或额外的 Git worktree。
Placo、Pinocchio 和对应 cmeel ABI 版本也在锁文件中，重新安装时应保留这套锁定组合。
运行官方手柄 IK、数据转换或 ACT 时需要安装 LeRobot；上述命令会一并完成安装，
无需另外克隆或手动安装一个 LeRobot 仓库。`.venv` 由对方本机创建，不随 Git 仓库分发。

## 启动

### F310 手柄控制仿真

```bash
./.venv/bin/python scripts/teleoperate_gamepad.py --ik official
```

手柄使用 D 模式；按住 **RB** 操作，松开后保持参考目标。
左摇杆控制 XY，右摇杆上下控制 Z，LT 关闭、RT 打开夹爪。
Esc、关闭窗口或 Ctrl-C 退出。此入口只控制本机仿真。

macOS 的普通 Python 负责输入，脚本自动启动独立 `mjpython` Viewer 子进程。
F310 的 HID 报文补零适配与具体按键说明见 [手柄文档](docs/gamepad_mujoco.md)。

### 查看场景

```bash
./.venv/bin/mjpython scripts/view_so101.py
```

加载 `models/so101/scene_fixed.xml`，从 `folded_ready` keyframe 开始。

### 专家抓取、放入盒内并返回

```bash
./.venv/bin/mjpython scripts/launch_pick_place.py --profile simulation_fast
```

无窗口执行或指定方块位置：

```bash
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --profile real_safe
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --cube-xy 0.22 0.05 --cube-yaw-deg 30
```

`simulation_fast` 为 60 Hz 仿真调试配置，`real_safe` 为 30 Hz 仿真配置。
这些配置名称不代表已在真实机械臂上验证。

## 手柄与 IK

| `--ik` 参数 | 行为 |
| --- | --- |
| `official`（默认） | 原版 LeRobot 0.6.1 游戏手柄运动学处理链与默认 IK 参数 |
| `dls-position` | 旧本地 DLS 位置对照，保留旧入口的运动策略 |
| `dls-grasp` | 旧抓取 DLS，包含末端朝下与 yaw=0 约束，保留旧入口的运动策略 |

官方模式的任务参数集中在 [config/gamepad_official.json](config/gamepad_official.json)：

| 参数 | 值与来源 |
| --- | --- |
| 控制频率 | 30 Hz，官方环境默认值 |
| URDF / 工具坐标 | 原始 `so101_new_calib.urdf` / `gripper_frame_link` |
| IK 位置 / 姿态权重 | 官方默认 `1.0 / 0.01` |
| XYZ 任务框 | 各 `[-1, 1]` 米，用户选定的官方通用示例范围 |
| XYZ 输入步长 | 各 `0.02` 米，官方 HIL IK 配置示例值 |

任务框是输入限幅范围，实际可达性由模型和 IK 决定。
官方模式不使用旧抓取地图作为三维限制；保留原版限幅和跨帧检查。
详细参数来源见 [官方处理链说明](docs/gamepad_mujoco.md)。

## 采集与推理

采集到新的批次目录：

```bash
./.venv/bin/python scripts/collect_pick_place_dataset.py \
  --episodes 20 --profile real_safe --output-dir datasets/raw_new
```

默认记录 front、wrist 两路 320×240 图像，`real_safe` 为 30 Hz。
采集器检查 Git 工作区；未提交修改时默认拒绝采集。
调试选项 `--allow-dirty` 会写入元数据，正式转换拒绝这样的样本。
同一目录重复运行会续采。数据结构见 [格式说明](datasets/raw/README.md)。

已有原始数据后，先检查再转换：

```bash
./.venv/bin/python scripts/convert_to_lerobot.py --dry-run --raw-dir datasets/raw_new
./.venv/bin/python scripts/convert_to_lerobot.py \
  --raw-dir datasets/raw_new --repo-id local/so101_pickplace \
  --root datasets/processed/so101_pickplace --verify
```

ACT 推理入口保留，但**本仓库未附训练权重**。取得完整 `pretrained_model/` 目录
（包含模型、配置和预处理器文件）后指定路径：

```bash
./.venv/bin/mjpython scripts/interactive_infer.py --checkpoint /path/to/pretrained_model
```

当前实现需要 front、wrist 双相机图像及 6 维关节状态。
单路顶部实机相机不能直接替代这两路输入；无窗口模式使用占位图像，只能检查加载和动作输出。

## 相机

HF868 的 640×480 内参示例保存在
[相机配置目录](references/project_metadata/configs/camera/hf868_135_640x480/)，
加载工具为 [camera_model.py](src/so_ws/common/camera_model.py)。
`scene_hf868.xml` 是追加顶部相机的叠加场景，其外参仍是临时估算值。
[外参 JSON](results/extrinsics_hf868.json) 是复现相机场景测试所需的小文件，不是采集数据集。

## 验证

基础模型检查与完整项目测试：

```bash
./.venv/bin/python -m so_ws.scripts.verify_model
./.venv/bin/python -m pytest tests -q
```

测试不依赖历史采集数据或训练权重；相机渲染测试需要可用的离屏图形上下文。
本次交付的验证结果和范围见 [开发版交付说明](docs/development_delivery.md)。
修改模型或控制行为后，按 [测试说明](tests/README.md) 选择相关回归检查。

## 目录与资料

```text
config/                    官方手柄模式任务配置
models/so101/              MJCF、URDF、STL 和相机叠加场景
src/                       控制器、专家策略、采样器与 MuJoCo 适配
scripts/                   场景、手柄、抓取、采集、转换和推理入口
tests/                     单元测试与离线集成测试
datasets/                  格式说明和空目录标记；实际数据被忽略
references/                相机配置示例
results/                   仅保留测试所需的外参 JSON
third_party/lerobot_v061/   带许可证的官方源码快照
docs/                      使用说明、技术参考和交付记录
```

[手柄使用说明](docs/gamepad_mujoco.md) · [技术参考与历史测量](docs/project_reference.md) ·
[第三方来源与许可](THIRD_PARTY_NOTICES.md)

技术参考中的历史批次、测试结果和设备调查用于解释设计，不表示相应数据、设备或权重随本仓库提供。
