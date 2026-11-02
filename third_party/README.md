# 第三方源码

`lerobot_v061/` 包含官方 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
0.6.1 的源码快照，固定提交为：

```text
7e241bd630a3719a56157a497ce5d08f244784f1
```

保留原提交的 1006 个跟踪文件、内部符号链接及 Apache-2.0 许可证，不改写官方源码。
原开发机可以保留原有 Git worktree，但 `.git` 指针只留在本机，不上传到仓库。
普通克隆即包含源码，按相对路径 editable 安装，无需原机器路径或额外的上游仓库。

自有 MuJoCo 适配放在 `src/so_ws/`。手柄报文兼容与单位桥接见
[手柄说明](../docs/gamepad_mujoco.md)，统一安装命令见 [项目首页](../README.md#安装与环境)。
