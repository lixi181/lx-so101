# 测试目录

测试按功能划分，可以离线运行；游戏手柄、官方 IK 及转换相关测试需要安装随仓库提供的 LeRobot 和 `gamepad,dev` 依赖。
在项目根目录执行 `uv sync --frozen --python 3.12 --extra gamepad --extra dev` 即可安装。
测试不依赖历史采集数据或训练权重；相机渲染测试需要可用的离屏图形上下文。

- `test_model.py`：场景与模型加载。
- `test_folded_ready.py`：起始 keyframe 姿态。
- `test_square_grasp_planner.py`：yaw 候选、世界 yaw 选择和 minimum-jerk 端点。
- `test_pick_cube.py`：固定场景下的抓取、抬升、放置和回程。
- `test_action_alignment.py`：开环回放证明 `(obs, action)` 严格同拍；记录的动作必须是
  限速后真正执行的命令。
- `test_yaw_alignment.py`：`solve_ik` 的 yaw 约束，固定 (0.25, 0) 时 0–75° 全部成功；
  同时验证不加约束时夹爪朝向与方块 yaw 无关。
- `test_task_sampling.py`：采样范围、可复现性、`apply_task` 的写入结果和种子推导。
- `test_reachability.py`：执行前的可达性预检，包括折叠臂穿模样本的拒绝。
- `test_collector.py`：采集器的原子写入、失败归档、续采种子、临时目录清理和元数据；
  物理部分用假的 `run()` 替换，只验证管道。

`module_loader.py` 提供 `load_script()`：`scripts/` 不是包，测试用它按路径加载脚本，
并把脚本目录加进 `sys.path`（与 `python scripts/xxx.py` 的实际运行方式一致）、注册到
`sys.modules`（`dataclasses` 解析字符串注解需要）。

运行全部测试：

```bash
./.venv/bin/python -m pytest tests/ -q
```
