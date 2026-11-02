# 第三方来源与许可

| 内容 | 上游与版本 | 许可文件 |
| --- | --- | --- |
| `third_party/lerobot_v061/` | [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，提交 `7e241bd630a3719a56157a497ce5d08f244784f1` | [Apache-2.0](third_party/lerobot_v061/LICENSE) |
| `models/so101/` 中的 SO-101 模型与网格 | [TheRobotStudio/SO-ARM100 / Simulation/SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) | [Apache-2.0](models/so101/LICENSE) |

SO-101 模型经过本工作空间适配，包含任务场景、夹爪接触配置、固定初始姿态和相机叠加场景；
具体行为以随仓库分发的 XML/URDF 为准。模型来源说明保留在 [模型 README](models/so101/README.md)。

项目的设计与方法参考见 [技术参考](docs/project_reference.md) 中的原始链接；
引用其他项目不表示这些项目的完整源码随本仓库分发。
本次发布未给自有代码新增开源许可证；第三方文件保留各自原有许可。
