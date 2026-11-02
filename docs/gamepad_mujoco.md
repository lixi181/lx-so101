# 本机 F310 → 官方 LeRobot IK → MuJoCo

默认启动模式为 `official`。本入口仅控制本机 MuJoCo，不打开机械臂串口、相机或远端连接。
真实 F310 的只读输入和报文兼容已经验证；本轮官方处理链接入只做离线测试，没有打开窗口。
LeRobot 源码随仓库提供；安装命令见 [项目首页](../README.md#安装与环境)。
下文引用的历史 results/ 报告未随开发版分发，交付验证见 [开发版说明](development_delivery.md)。

## 启动

在项目根目录执行：

```bash
./.venv/bin/python scripts/teleoperate_gamepad.py --ik official
```

省略 `--ik` 也使用 `official`。先关闭旧入口，再运行新命令，正在运行的旧进程不会自动更新代码。
终端、输入窗口和 Viewer 子进程握手均记录实际策略。

按住 RB 操作，松开 RB 保持参考目标；左摇杆控制 XY，右摇杆上下控制 Z，LT 关闭、RT 打开夹爪。
Esc、关闭任意窗口或终端 Ctrl-C 退出。没有添加新的回中重接管门槛。

## 官方代码与配置来源

LeRobot 源码位于 `third_party/lerobot_v061/`，提交
`7e241bd630a3719a56157a497ce5d08f244784f1`，官方源码未修改。

实际运动学子链使用官方 `src/lerobot/rl/gym_manipulator.py` 中的类和顺序：

1. `MapDeltaActionToRobotActionStep`，默认缩放及噪声阈值。
2. `EEReferenceAndDelta`，`use_latched_reference=False`、`use_ik_solution=True`。
3. `EEBoundsAndSafety`，工作区限幅；保留其默认 0.05 m 跨帧检查和 `raise_on_jump=True`。
4. `GripperVelocityToJoint`，`clip_max=100`、`speed_factor=1.0`、`discrete_gripper=True`。
5. `InverseKinematicsRLStep`，`initial_guess_current_joints=False`。

以上参数与官方游戏手柄链路相同。输入已经是官方 GamepadTeleop 的字典，因此无需
策略张量转换步骤；输出直接转换成 MuJoCo 执行器单位，无需策略张量输出步骤。
不运行训练、奖励分类器、数据采集或任何硬件环境。

任务参数集中在 `config/gamepad_official.json`，不是整个 LeRobot 的统一默认值：

| 项目 | 当前值 | 来源 |
| --- | --- | --- |
| 控制频率 | 30 Hz | 官方环境默认及通用遥操作示例 |
| URDF | `so101_new_calib.urdf` | 官方推荐的原始 SO-101 模型 |
| 工具坐标 | `gripper_frame_link` | 官方默认及遥操作示例 |
| XYZ 任务框 | 各 `[-1, 1]` m | 用户明确选择的官方通用示例范围 |
| XYZ 输入步长 | 各 0.02 m | 官方 `docs/source/hilserl.mdx` 的 IK 配置示例 |
| 夹爪归一化上限 | 100 | 官方环境默认值 |
| IK 位置/姿态权重 | 1.0 / 0.01 | `RobotKinematics.inverse_kinematics` 原版默认值 |

步长是每次输入的位移比例，不是之前自定的 0.03 m/s 速度。实际处理遵循官方步骤。
官方本身要求任务提供工作区和位移步长，因此这里明确记录了各参数的具体来源。

没有追加旧入口中的 12～34 cm 范围、2 mm 残差拒绝、100 次 IK 循环、虚拟 TCP、
朝下/yaw 锁定、关节变化裁剪、速度/加速度整形、碰撞触发全局暂停或 250 ms 超时规则。
每份输入经过原版 IK 一次，没有新输入时只推进已有仿真目标，不重复积分旧输入。
官方限幅步骤不会因单纯触边锁住全部控制；其自身跨帧突跳错误仍按原版行为抛出。

## 必需的设备与仿真桥接

- F310 以 D 模式 `046D:C216 / Logitech Dual Action` 连接。
- 当前 macOS 的 pygame/SDL 无法枚举它，因此使用官方 `hidapi_fallback=True`。
- F310 的实际 8 字节报文缺少官方 RumblePad 解析器预期的前导 report ID；
  本地只补一个零字节，再交给官方解析器，官方源文件未修改。
- 官方运动学使用度数，MuJoCo 使用弧度，桥接执行单位转换。
- 官方 SO100 离散夹爪处理器以归一化数值增大表示关闭，当前 MJCF 的正角方向表示打开，
  桥接按模型原有端点转换夹爪方向，没有更改官方夹爪处理器。
- 使用场景自己的 `folded_ready` 初始化，不调用本地抓取 IK 自动展开到指定工作点。
- 原 URDF/XML 和 MuJoCo 动力学不变；碰撞继续由模型物理引擎处理。

主入口是 `scripts/teleoperate_gamepad.py`，官方仿真桥接为
`src/so_ws/teleoperators/official_gamepad_mujoco.py`。用户选定的任务框不等于机械臂实际可达范围。

## 工作地图核查

`results/collection_maps/collection_region_map.json` 记录的是 raw_v1/raw_v2 方块放置 XY：
名义半径分别 0.20～0.24 m、0.20～0.26 m，方位角 ±30°。
`workspace_calibration_*.json` 验证特定抓放任务及方块朝向。
V5 备份说明记录的 576 个 1 cm 网格也是桌面 XY 有限任务区域。
这些资料均未给出通用 TCP 的 Z 上下限，因此没有用它们给官方入口追加三维边界。
核查记录在 `results/official_ik_20261025/workspace_audit.json`。

## 旧模式和记录

`--ik dls-position`、`--ik dls-grasp` 保留为旧本地对照，仍有旧入口的运动策略，
不属于本次要求的官方配置。此前的 `placo-position`、`placo-soft` 不再作为启动选项，
避免把调参对照模式误当成原版官方接口。

历史备份与报告保留于 `results/gamepad_integration_20261025/`、
`results/ik_modes_20261025/`、`results/ik_comparison_20261025/`。
本轮备份和验证记录位于 `results/official_ik_20261025/`。

本轮 101 项离线测试、Ruff 及锁文件检查通过；官方原有 1006 个跟踪文件校验一致。
详细结果见 `results/official_ik_20261025/RESULTS.md`。新官方模式的窗口交互仍待实际启动验证。
