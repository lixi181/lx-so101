"""手柄仿真可选 IK；不修改原抓取求解器或官方 LeRobot 源码。"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from controllers.ik_controller import ARM_JOINTS, GRASP_OFFSET, grasp_point, solve_ik

LEGACY_IK_MODES = {
    "placo-position": "旧 Placo 位置对照模式",
    "placo-soft": "旧 Placo 软姿态对照模式",
    "dls-position": "本地 DLS：只约束位置",
    "dls-grasp": "原抓取 DLS：朝下 + yaw=0",
}
IK_MODES = {
    "official": "官方 LeRobot 0.6.1 游戏手柄处理链（±1 米任务框）",
    "dls-position": LEGACY_IK_MODES["dls-position"],
    "dls-grasp": LEGACY_IK_MODES["dls-grasp"],
}
DEFAULT_IK = "official"


class GamepadIK:
    def __init__(self, mode, model, data, qids, dofs, gripper_id):
        if mode not in LEGACY_IK_MODES:
            raise ValueError(f"未知 IK 模式：{mode}")
        self.mode = mode
        self.model, self.data = model, data
        self.qids, self.dofs, self.gripper_id = qids, dofs, gripper_id
        self.ranges = np.array([model.jnt_range[model.joint(name).id] for name in ARM_JOINTS])
        self.scratch = mujoco.MjData(model)
        self.kinematics = None
        self.reference_pose = self._pose(data.qpos[qids])
        if mode.startswith("placo-"):
            self._load_placo()

    def _pose(self, q):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.scratch)
        transform = np.eye(4)
        transform[:3, :3] = self.scratch.xmat[self.gripper_id].reshape(3, 3)
        transform[:3, 3] = grasp_point(self.model, self.scratch, self.gripper_id)
        return transform

    def _load_placo(self):
        try:
            from lerobot.model.kinematics import RobotKinematics
        except (ImportError, OSError) as error:
            raise RuntimeError("官方 IK 依赖不可用，请安装项目 gamepad 依赖；不会自动切换 IK") from error
        source = Path(__file__).resolve().parents[3] / "models/so101/so101_new_calib_camera.urdf"
        tree = ET.parse(source)
        root = tree.getroot()
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.set("filename", str(source.parent / filename))
        # 统一为现有 MuJoCo 抓取点，不使用相差约 25.48 mm 的默认 gripper_frame_link。
        ET.SubElement(root, "link", {"name": "gamepad_tcp"})
        joint = ET.SubElement(root, "joint", {"name": "gamepad_tcp_joint", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "gripper_link"})
        ET.SubElement(joint, "child", {"link": "gamepad_tcp"})
        ET.SubElement(joint, "origin", {"xyz": " ".join(map(str, GRASP_OFFSET)), "rpy": "0 0 0"})
        with tempfile.TemporaryDirectory(prefix="so101_gamepad_ik_") as directory:
            path = Path(directory) / "robot.urdf"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            self.kinematics = RobotKinematics(str(path), "gamepad_tcp", list(ARM_JOINTS))
        solver = self.kinematics
        solver.solver.mask_dof("gripper")
        solver.solver.enable_joint_limits(True)
        for index, name in enumerate(ARM_JOINTS):
            solver.robot.set_joint_limits(name, *self.ranges[index])
        q = self.data.qpos[self.qids].copy()
        alignment = self.reference_pose @ np.linalg.inv(solver.forward_kinematics(np.rad2deg(q)))
        solver.robot.set_T_world_fbase(alignment)
        # 多个姿态检查模型一致性，防止基座对齐掩盖关节符号或模型版本不一致。
        for offset in [np.zeros(5), *np.eye(5) * 0.1]:
            trial = np.clip(q + offset, self.ranges[:, 0], self.ranges[:, 1])
            error = np.linalg.norm(
                solver.forward_kinematics(np.rad2deg(trial))[:3, 3] - self._pose(trial)[:3, 3]
            )
            if error > 0.00002:
                raise RuntimeError(f"Placo 与 MuJoCo 的 TCP 模型不一致：{error * 1000:.4f} mm")

    def solve(self, target, seed):
        if self.mode == "dls-grasp":
            return solve_ik(
                self.model, self.data, target, self.qids, self.dofs, self.gripper_id, target_yaw=0.0
            )
        if self.mode == "dls-position":
            q = seed.copy()
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            for _ in range(200):
                point = self._pose(q)[:3, 3]
                error = target - point
                if np.linalg.norm(error) < 0.0002:
                    break
                mujoco.mj_jac(self.model, self.scratch, jacp, jacr, point, self.gripper_id)
                jac = jacp[:, self.dofs]
                q += 0.65 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(3), error)
                q = np.clip(q, self.ranges[:, 0], self.ranges[:, 1])
        else:
            q = seed.copy()
            desired = self.reference_pose.copy()
            desired[:3, 3] = target
            weight = 0.01 if self.mode == "placo-soft" else 0.0
            previous_error, stalled = float("inf"), 0
            for _ in range(100):
                self.kinematics.forward_kinematics(np.rad2deg(q))
                q = np.deg2rad(
                    self.kinematics.inverse_kinematics(np.rad2deg(q), desired, orientation_weight=weight)
                )
                error = float(np.linalg.norm(self._pose(q)[:3, 3] - target))
                if error < 0.0002:
                    break
                stalled = stalled + 1 if abs(previous_error - error) < 1e-8 else 0
                if stalled >= 5:
                    break
                previous_error = error
        return q, float(np.linalg.norm(self._pose(q)[:3, 3] - target))
