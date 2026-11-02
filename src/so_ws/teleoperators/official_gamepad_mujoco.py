"""官方游戏手柄运动学处理链到 MuJoCo 的单位/进程适配。

处理器顺序和参数来自 LeRobot 0.6.1 rl/gym_manipulator.py。
任务参数来自 config/gamepad_official.json，来源逐项记录在操作说明中。
不使用此前的自定义任务范围、虚拟 TCP、额外求解循环或运动限制。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import DataProcessorPipeline, MapDeltaActionToRobotActionStep, TransitionKey
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)

from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config/gamepad_official.json"


class OfficialGamepadMujoco:
    def __init__(self):
        self.config = json.loads(CONFIG_PATH.read_text())
        self.ik_mode = "official"
        self.model, self.data = load_model(FIXED_SCENE_PATH)
        self.dt = 1.0 / self.config["fps"]
        self.substeps = round(self.dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.dt):
            raise ValueError("官方控制周期与 MuJoCo 物理步长不匹配")
        self.qids = [int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in JOINTS]
        self.aids = [self.model.actuator(name).id for name in JOINTS]
        self.gripper_range = self.model.jnt_range[self.model.joint("gripper").id].copy()
        # 仿真初始化使用场景自身的 keyframe，不调用本地抓取 IK 自动展开。
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("folded_ready").id)
        mujoco.mj_forward(self.model, self.data)
        self.target = self.data.qpos[self.qids].copy()
        self.data.ctrl[self.aids] = self.target
        config = self.config["inverse_kinematics"]
        root = CONFIG_PATH.parents[1]
        self.kinematics = RobotKinematics(
            urdf_path=str(root / config["urdf_path"]),
            target_frame_name=config["target_frame_name"],
            joint_names=list(JOINTS),
        )
        # 与官方游戏手柄 IK 子处理链一致，不修改各处理器源码或默认值。
        self.pipeline = DataProcessorPipeline(
            steps=[
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=self.kinematics,
                    end_effector_step_sizes=config["end_effector_step_sizes"],
                    motor_names=list(JOINTS),
                    use_latched_reference=False,
                    use_ik_solution=True,
                ),
                EEBoundsAndSafety(end_effector_bounds=config["end_effector_bounds"]),
                GripperVelocityToJoint(
                    clip_max=self.config["max_gripper_pos"],
                    speed_factor=1.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsRLStep(
                    kinematics=self.kinematics,
                    motor_names=list(JOINTS),
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=identity_transition,
            to_output=identity_transition,
        )
        self.pending_action = None
        self.complementary_data = {}
        self.enabled = False
        self.reason = "官方处理链就绪，按住 RB 操作"

    def observation(self):
        values = np.rad2deg(self.data.qpos[self.qids]).tolist()
        low, high = self.gripper_range
        # 官方 SO100 离散夹爪处理器使用数值增大表示闭合；当前 MJCF 正角方向为打开。
        values[-1] = (high - self.data.qpos[self.qids[-1]]) / (high - low) * self.config["max_gripper_pos"]
        return {f"{name}.pos": float(value) for name, value in zip(JOINTS, values, strict=True)}

    def receive(self, packet, now):
        self.enabled = bool(packet["enabled"])
        self.pending_action = (
            dict(packet["action"])
            if self.enabled
            else {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1,
            }
        )
        self.reason = "官方处理链控制中" if self.enabled else "RB 已松开，保持官方参考目标"

    def advance(self, now):
        if self.pending_action is not None:
            transition = self.pipeline(
                create_transition(
                    observation=self.observation(),
                    action=self.pending_action,
                    complementary_data=self.complementary_data,
                )
            )
            self.complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
            action = transition[TransitionKey.ACTION]
            self.target[:5] = np.deg2rad([action[f"{name}.pos"] for name in JOINTS[:5]])
            low, high = self.gripper_range
            self.target[5] = high - action["gripper.pos"] / self.config["max_gripper_pos"] * (high - low)
            self.data.ctrl[self.aids] = self.target
            self.pending_action = None
        # 一份输入只经过官方处理链一次；没有新输入时只推进已有仿真目标。
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
