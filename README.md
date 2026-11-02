# SO-101 本机 MuJoCo 开发工作空间

本项目用于 SO-101 的 MuJoCo 仿真、F310 手柄遥操作、专家轨迹采集和 ACT 策略推理。
以下命令均在项目根目录执行；历史测量、批次结果保留在技术参考中。
仓库不包含采集数据、训练权重或虚拟环境。

常用导航：[启动方式](#启动方式) · [手柄与-ik-配置](#手柄与-ik-配置) ·
[数据采集与转换](#数据采集与转换) · [目录结构](#目录结构) · [文档与记录](#文档与记录)

## 安装与环境

本项目在 macOS Apple Silicon、Python 3.12.15 上验证，依赖声明见
[pyproject.toml](pyproject.toml)。新开发者先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

```bash
git clone https://github.com/lixi181/lx-so101.git
cd lx-so101
uv sync --frozen --python 3.12 --extra gamepad --extra dev
```

已有工作目录时，在该目录执行上面的 `uv sync` 即可；LeRobot 会一并安装。
依赖版本由 [uv.lock](uv.lock) 固定。macOS 上直接启动 MuJoCo 窗口使用 `mjpython`；
手柄入口使用普通 `python`，由脚本自行启动 Viewer 子进程。

| 内容 | 仓库与使用条件 |
| --- | --- |
| LeRobot | 0.6.1，位于 `third_party/lerobot_v061/`，以 editable 方式接入 |
| F310 默认策略 | `official`，直接使用官方运动学处理链 |
| 原始数据 | 不随仓库提供；原开发机保留 `raw_v3` 的 300 条轨迹，`raw_v2` 已清理 |
| 已转换数据 | `datasets/processed/` 仅含占位文件，转换结果不上传 |
| ACT checkpoint | 不随仓库提供；运行推理前需另行取得完整 `pretrained_model/` 目录 |
| 实机相机资料 | HF868 的 640×480 内参已存档；外参仍是临时估算值 |

LeRobot 0.6.1 的源码快照已随仓库提供，按相对路径以 editable 方式安装。
原开发机可以继续使用既有 Git worktree；克隆者无需原机器的目录或上游 Git 历史。
来源与许可证见 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 启动方式

### F310 手柄控制仿真

```bash
./.venv/bin/python scripts/teleoperate_gamepad.py --ik official
```

省略 `--ik` 也使用 `official`。先关闭旧入口窗口，再运行新命令。
按住 **RB** 操作，松开后保持参考目标；左摇杆控制 XY，右摇杆上下控制 Z，
LT 关闭、RT 打开夹爪。Esc、关闭窗口或 Ctrl-C 退出。

此入口只控制本机 MuJoCo。真实 F310 输入已验证；官方处理链接入的验证记录为离线测试，
其窗口交互尚未在该记录中验证。详见 [手柄操作说明](docs/gamepad_mujoco.md)。

### 只查看场景

```bash
./.venv/bin/mjpython scripts/view_so101.py
```

加载 `models/so101/scene_fixed.xml`，从场景的 `folded_ready` keyframe 开始。

### 专家抓取、放入盒内并返回

```bash
./.venv/bin/mjpython scripts/launch_pick_place.py --profile simulation_fast
```

`simulation_fast` 为 60 Hz 仿真调试配置；`real_safe` 为 30 Hz 仿真配置。
两者都在本地仿真中执行。无窗口运行或指定方块位置：

```bash
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --profile real_safe
./.venv/bin/python scripts/launch_pick_place.py --no-viewer --cube-xy 0.22 0.05 --cube-yaw-deg 30
```

任务阶段、成功判据和采样范围见 [抓取任务技术说明](docs/project_reference.md#抓取盒内放置和返回初始姿态)。

### ACT 交互式推理

```bash
./.venv/bin/mjpython scripts/interactive_infer.py --checkpoint /path/to/pretrained_model
```

`--checkpoint` 指向另行取得的模型目录，需包含权重、配置与预处理器文件。
脚本自动选择可用的 MPS 或 CPU。
窗口配有方块控制面板，设置位置并点击“开始任务”后执行。
当前 checkpoint 输入为 **front、wrist 双相机图像及 6 维关节状态**，不能直接替换为单路顶部实机相机。

仅检查模型加载和动作输出时可使用：

```bash
./.venv/bin/python scripts/interactive_infer.py --checkpoint /path/to/pretrained_model --no-viewer --device cpu --max-ticks 60
```

这个无窗口模式使用占位图像，不是视觉闭环抓取成功率评估。
可用 `--checkpoint` 指定模型，`--action-steps` 调整动作队列长度；默认连续执行 100 步后重新推理。

## 手柄与 IK 配置

| `--ik` 参数 | 处理方式 |
| --- | --- |
| `official`（默认） | 官方 LeRobot 0.6.1 游戏手柄运动学处理链与默认 IK 参数 |
| `dls-position` | 旧本地 DLS 位置对照模式，保留旧入口的运动策略 |
| `dls-grasp` | 旧抓取 DLS，包含末端朝下与 yaw=0 约束，保留旧入口的运动策略 |

官方模式配置集中在 [config/gamepad_official.json](config/gamepad_official.json)：

| 参数 | 当前值与来源 |
| --- | --- |
| 控制频率 | 30 Hz，官方环境默认值 |
| URDF / 工具坐标 | 原始 `so101_new_calib.urdf` / `gripper_frame_link` |
| IK 位置 / 姿态权重 | 官方默认 `1.0 / 0.01` |
| XYZ 任务框 | 各 `[-1, 1]` 米，用户选定的官方通用示例范围 |
| XYZ 输入步长 | 各 `0.02` 米，官方 HIL IK 配置示例值 |

任务框不是机械臂的实际可达范围。旧抓取地图没有用于限制官方遥操作；
原版 `EEBoundsAndSafety` 的限幅和跨帧检查仍保留。
F310 的 HID 报文兼容、单位转换及夹爪方向桥接见 [完整说明](docs/gamepad_mujoco.md)。

## 数据采集与转换

数据目录的批次名（如 `raw_v2`）与文件格式版本（`local_raw_v3`）是两件事。
当前采集器默认记录 front、wrist 两路 320×240 图像，`real_safe` 配置为 30 Hz，
动作是绝对关节位置目标，观测与下一控制 tick 的动作同拍。

采集到一个新的批次目录，例如：

```bash
./.venv/bin/python scripts/collect_pick_place_dataset.py \
  --episodes 20 --profile real_safe --output-dir datasets/raw_new
```

采集器默认检查 Git 工作区，未提交修改时会拒绝采集。
调试用的 `--allow-dirty` 会在元数据中留下标记；正式转换会拒绝这些样本。
同一目录再次采集会从已有编号后续写，指定输出目录前应确认是否要续采。

以下以采集得到的 `datasets/raw_new` 为例，先扫描已有数据：

```bash
./.venv/bin/python scripts/convert_to_lerobot.py --dry-run --raw-dir datasets/raw_new
```

实际转换并读回校验：

```bash
./.venv/bin/python scripts/convert_to_lerobot.py \
  --raw-dir datasets/raw_new \
  --repo-id local/so101_pickplace_v3 \
  --root datasets/processed/so101_pickplace_v3 --verify
```

转换只接受成功、Git 状态干净、格式为 `local_raw_v3`、30 Hz、320×240 的 episode。
LeRobot 已接入本机；转换是否完成以输出文件和 `--verify` 结果为准。
格式见 [原始数据说明](datasets/raw/README.md)，历史试点、分层采样、评估布局和单位映射见
[数据技术参考](docs/project_reference.md#本地数据集采集)。

## 相机与场景

HF868 内参镜像位于
[references/project_metadata/configs/camera/hf868_135_640x480/](references/project_metadata/configs/camera/hf868_135_640x480/)，
对应 640×480、`fx=326.68129`、`fy=326.27539`。
加载接口是 [camera_model.py](src/so_ws/common/camera_model.py)，默认登记见
[cameras.yaml](references/project_metadata/configs/cameras.yaml)。

`models/so101/scene_hf868.xml` 在原固定场景上追加顶部相机。
其中外参是临时估算值，原记录估算的不确定度约 ±14.5 mm，不作为精细抓取标定结果。
标定来源、投影约定和后续复标记录见 [相机技术参考](docs/project_reference.md#真实相机内参)。
HF868 顶部相机与 ACT checkpoint 使用的 front、wrist 相机配置分别维护。

`scene_fixed.xml` 的哈希被历史数据与实验记录引用；调整相机使用叠加场景，避免改写历史数据的场景依据。

## 目录结构

```text
config/                    当前手柄官方模式任务配置
models/so101/              MJCF、URDF、网格与相机叠加场景
src/controllers/           本地 IK、抓取规划与运动时序
src/experts/               专家策略
src/so_ws/                 MuJoCo 适配、相机工具与遥操作桥接
src/task_sampling.py       任务初始状态采样
scripts/                   Viewer、手柄、抓取、采集、转换和推理入口
tests/                     单元测试与离线集成测试
datasets/                  原始批次、转换数据、评估布局
outputs/                   ACT checkpoint 等模型产物
experiments/               实验配置、脚本及记录
results/                   标定、地图、验证报告与历史结果
references/                外部资料与相机标定镜像
third_party/lerobot_v061/   LeRobot 源码快照；原开发机可保留既有 Git worktree
docs/                      使用说明和详细技术参考
md/                        独立设计方案
remote_staging/            远端部署相关材料
```

`results/`、`outputs/` 等本机产物不等同于 Git 已跟踪文件，迁移时需单独核对。
远端部署材料按各自目录的说明使用；上面的启动命令均针对本机仿真。

## 检查与验证

模型基础检查：

```bash
./.venv/bin/python -m so_ws.scripts.verify_model
```

相关离线测试：

```bash
./.venv/bin/python -m pytest \
  tests/test_model.py tests/test_gamepad_ik_modes.py tests/test_official_gamepad_mujoco.py -q
```

2026-11-02 的精简开发版验证为 221 项测试通过、1 项跳过，详见
[交付验证说明](docs/development_delivery.md)。历史实验报告不随仓库提供。

## 文档与记录

| 文档 | 内容 |
| --- | --- |
| [手柄操作说明](docs/gamepad_mujoco.md) | F310 按键、官方处理链、参数来源与兼容适配 |
| [项目技术参考与历史记录](docs/project_reference.md) | 相机标定、抓取判据、数据格式、历史试点与采样结果 |
| [开发版交付说明](docs/development_delivery.md) | 分发范围、依赖安装、源码校验与离线验证结果 |
| [原始数据格式](datasets/raw/README.md) | episode 文件结构与观测/动作语义 |

模型来源：[SO-ARM100 / SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)。
基础框架：[LeRobot](https://github.com/huggingface/lerobot)。
职责划分参考：[SO-101 SimStudio 设计](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/DESIGN.md)。
