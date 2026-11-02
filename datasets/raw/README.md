# 原始数据说明

本文件是**各批次共用的格式说明**。SO-101 专家演示的原始轨迹格式版本为 `local_raw_v3`，
由 `scripts/collect_pick_place_dataset.py` 生成，不依赖 LeRobot。

本目录 `datasets/raw/` 自身不再保存数据（原来那 3 条 `local_raw_v2` 试采已删除，只留这份
说明和占位文件）。实际数据按批次分开存放，格式都一样：

| 目录 | 内容 |
|---|---|
| `datasets/raw_v1/` | 第一版正式批次（分层采样，300 条或 200 条，待采） |
| `datasets/raw_v1_pilot/` | 第一版试点批次（50 条，已完成） |
| `datasets/raw/` | 已清空；只保留本文件 |

`datasets/raw*` 与 `datasets/processed`、`datasets/evaluation` 下的内容都不进 Git。

## 目录结构

```text
<output-dir>/                  例如 datasets/raw_v1/
  index.jsonl                 每个 episode 一行：编号、种子、成功与否、失败阶段和原因
  manifest.json               本批采集的命令行、起止时间、汇总和版本信息
  episode_000000/             成功的 episode
    episode.json
    state_action.npz
    front_camera/*.png
    wrist_camera/*.png
  failed/
    episode_000003/           失败的 episode，数据同样完整，可作负样本或失败分析
  .tmp_episode_000007/        采集中的临时目录，正常结束时原子改名，异常残留下次启动时清理
```

成功和失败分开存放：`episode_*` 是成功轨迹，`failed/episode_*` 是失败轨迹。
写入流程是「先写 `.tmp_episode_XXXXXX`，再 rename 到最终目录」，同一文件系统内的 rename
是原子操作，所以中途 Ctrl-C 不会留下半个 episode。

## 动作语义（v2 与 v1 的关键区别）

```json
"action_semantics": "absolute_joint_position_target, applied over next tick with linear interpolation",
"obs_action_alignment": "obs[t] recorded before action[t] is applied"
```

`state_action.npz` 里的 `action[t]` 是 6 维绝对关节位置目标（5 个手臂关节 + 夹爪），
在下一个控制 tick（`real_safe` 为 1/30 s ≈ 33.3 ms）内从上一帧的命令线性插值执行。观测在同一
tick 的物理步之前记录，因此 `qpos[t]` 与 `action[t]` 严格同拍：`qpos[t]` 加上
`action[t]` 才是 `qpos[t+1]` 的成因，初始状态那一帧也会被记录。

v1 记录的是「导致当前状态的命令」，即 `(s_t, a_{t-1})` 错了一拍。v2 修正了这个问题，
并在 `advance_to_ctrl`/`step` 中强制物理步数是控制 tick 的整数倍。两个版本的数据不能
混用，训练脚本必须按 `format` 字段区分。

`tests/test_action_alignment.py` 用开环回放验证对齐：按记录的动作序列回放，每个 tick
开始时的 `qpos` 必须与记录的 `qpos[t]` 一致（容差 1e-9）。动作错一拍这个测试一定失败。

## 字段说明

`state_action.npz`：

| 字段 | 形状 | 含义 |
|---|---|---|
| `timestamp` | `(T,)` | MuJoCo 时间，单位秒 |
| `qpos` | `(T, 6)` | 5 个手臂关节 + 夹爪位置 |
| `qvel` | `(T, 6)` | 对应速度 |
| `action` | `(T, 6)` | 限速后的绝对关节位置目标，见上文动作语义 |
| `cube_pos` / `cube_quat` | `(T, 3)` / `(T, 4)` | 方块位姿 |
| `phase` | `(T,)` | 阶段标签（接近/下降/gripper/抬升/…），便于分段分析 |

`episode.json` 中的关键字段：

- `format`：`local_raw_v3`（30 Hz、采集默认 320×240）。v1 记录的动作错了一拍；v2 修正了
  对齐但跑在 31.25 Hz / 160×120 上；v3 把 `timestep` 改成 1/480 s，`real_safe` 控制周期
  恰好 1/30 s，并把默认分辨率提到 320×240。三个版本的动作语义相同，不能混用：训练脚本
  要按 `format` 区分，或按 `sampling_hz` 和 `image_size` 判断。
- `git_commit` / `git_dirty`：采集时的代码版本；`git_dirty=true` 表示工作区有未提交改动。
- `mujoco_version` / `numpy_version` / `python` / `platform`：运行环境。
- `scene_sha256` / `scene_files`：`scene_fixed.xml` 及其 `include` 的 XML 哈希。
- `motion_profile`：完整的速度配置（`dataclasses.asdict`）。
- `control_period_sec`：由 `physics_per_tick × timestep` 算出，不再写死；`real_safe` 为
  1/30 s（`sampling_hz` 为精确的 30.0）。
- `sampler_mode`：`stratified` 或 `uniform`，顶层字段便于直接筛选。
- `sampler`：完整采样配置（含 `mode`、`bins`）、本 episode 的采样结果 `sample`（含分层
  采样的格子编号 `cell`，uniform 时为 `null`）、以及为拿到可达样本丢弃的 `rejected_samples`。

## 种子与续采

种子由编号推导：`derive_seed(base_seed, episode_index, attempt)`，同一个编号永远对应同一个
任务，与分几次采集无关。不指定 `--start-index` 时从 `max(已有编号)+1` 开始，续采不会覆盖、
也不会重复种子。

采集前会检查 `git status --porcelain`：工作区不干净时拒绝采集，除非显式传 `--allow-dirty`
（此时 `episode.json` 里 `git_dirty=true`，仍然可以反查 commit）。

## 采样范围

默认采样范围为 `r ∈ (0.20, 0.24) m`、`θ ∈ (-30°, 30°)`、方块 yaw ∈ `[0°, 90°)`，
由 5×5×4 网格标定得到，标定方法和完整结论见仓库根目录 `README.md`。
在 r ≤ 0.18 m 的格子上专家出现过「抬升阶段没夹住方块」的失败，因此默认范围排除了这一区域。

采集默认在这个范围上做**分层采样**（`--sampler-mode stratified`，r/θ/yaw 各 4 档共 64 格，
按编号轮转、格内抖动），丢弃重采仍在同一格内进行。要回到完全均匀随机用
`--sampler-mode uniform`。详见仓库根目录 `README.md` 的「分层采样」。
