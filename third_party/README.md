# 第三方源码

`lerobot_v061/` 是官方 [huggingface/lerobot](https://github.com/huggingface/lerobot)
的源码快照，版本 0.6.1，固定提交：

```text
7e241bd630a3719a56157a497ce5d08f244784f1
```

快照保留该提交的 1006 个跟踪文件、内部符号链接及 Apache-2.0 许可证。
没有复制上游的 `.git`、本机 worktree 指针、虚拟环境或缓存，也没有改写官方源码。
`pyproject.toml` 和 `uv.lock` 使用项目内的相对路径，`uv sync` 将其安装为 editable 依赖。
普通 `git clone` 即包含此快照，无需初始化 submodule 或重建原机器目录。

本项目的手柄报文兼容和 MuJoCo 单位桥接位于 `scripts/teleoperate_gamepad.py` 与
`src/so_ws/teleoperators/`，参数来源见 [手柄说明](../docs/gamepad_mujoco.md)。
