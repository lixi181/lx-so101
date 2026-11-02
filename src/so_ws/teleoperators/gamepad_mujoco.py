"""官方 GamepadTeleop 动作到本地 MuJoCo 的适配；不包含实臂后端。"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.ik_controller import arm_addresses, grasp_point, solve_ik
from so_ws.robots.so101_mujoco.model import FIXED_SCENE_PATH, load_model
from so_ws.teleoperators.gamepad_ik import LEGACY_IK_MODES, GamepadIK

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SUBSTEPS = 16
INPUT_TIMEOUT = 0.25
TCP_SPEED = 0.03
TCP_ACCELERATION = 0.10
JOINT_SPEED = 0.35
GRIPPER_SPEED = 0.60
WORKSPACE_MIN = np.array([0.12, -0.22, 0.02])
WORKSPACE_MAX = np.array([0.34, 0.22, 0.30])


class GamepadMujoco:
    """XYZ 增量 + 三态夹爪；显式选择 IK，类默认保留原抓取模式。

    手柄输出是归一化方向，必须乘速度和控制周期，不能直接当作米或关节角。
    启动、断流或非法输入后，需要先松开 RB 并回中，再按住 RB 接管。
    """

    def __init__(self, ik_mode="dls-grasp"):
        if ik_mode not in LEGACY_IK_MODES:
            raise ValueError(f"未知 IK 模式：{ik_mode}")
        self.ik_mode = ik_mode
        self.model, self.data = load_model(FIXED_SCENE_PATH)
        self.dt = SUBSTEPS * float(self.model.opt.timestep)
        self.arm_qids, self.arm_dofs = arm_addresses(self.model)
        self.qids = [int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in JOINTS]
        self.aids = [self.model.actuator(name).id for name in JOINTS]
        self.ranges = np.array([self.model.jnt_range[self.model.joint(name).id] for name in JOINTS])
        self.gripper_id = self.model.body("gripper").id
        self.cube_id = self.model.body("cube").id
        root_id = int(self.model.jnt_bodyid[self.model.joint("shoulder_pan").id])
        self.moving_bodies = self._descendants(root_id)
        self.gripper_bodies = self._descendants(self.gripper_id)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("folded_ready").id)
        mujoco.mj_forward(self.model, self.data)
        arm, error = solve_ik(
            self.model,
            self.data,
            np.array([0.22, 0.0, 0.08]),
            self.arm_qids,
            self.arm_dofs,
            self.gripper_id,
            target_yaw=0.0,
        )
        if error > 0.002:
            raise ValueError(f"仿真初始工作姿态不可达：{error:.4f} m")
        # 仅设置新仿真的初始状态，不执行自动展开或抓取，不改写原 XML。
        self.data.qpos[self.arm_qids] = arm
        self.data.qpos[self.qids[5]] = 1.20
        self.target = self.data.qpos[self.qids].copy()
        self.data.ctrl[self.aids] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.preview = mujoco.MjData(self.model)
        if self._collision(self.data):
            raise ValueError("仿真初始工作姿态发生碰撞")
        self.goal_xyz = grasp_point(self.model, self.data, self.gripper_id)
        self.velocity = np.zeros(3)
        self.action = np.zeros(3)
        self.gripper = 1
        self.enabled = False
        self.can_enable = False
        self.last_stamp = None
        self.last_seq = -1
        self.reason = "先松开 RB 并将摇杆回中，再按住 RB"
        self.ik = GamepadIK(ik_mode, self.model, self.data, self.arm_qids, self.arm_dofs, self.gripper_id)

    def _descendants(self, root):
        bodies = {root}
        for body in range(root + 1, self.model.nbody):
            if int(self.model.body_parentid[body]) in bodies:
                bodies.add(body)
        return bodies

    def _collision(self, data):
        for contact in data.contact:
            if contact.dist >= -0.0005:
                continue
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if not ({b1, b2} & self.moving_bodies):
                continue
            # 夹爪接触方块是任务的一部分；其他臂体碰撞均暂停。
            if (b1 == self.cube_id and b2 in self.gripper_bodies) or (
                b2 == self.cube_id and b1 in self.gripper_bodies
            ):
                continue
            return True
        return False

    def pause(self, reason):
        self.enabled = False
        self.can_enable = False
        self.velocity[:] = 0
        self.reason = reason

    def receive(self, packet, now):
        """只接收同主机、新鲜且递增的输入；非法包暂停并报错。"""
        try:
            seq, stamp, enabled = packet["seq"], packet["stamp"], packet["enabled"]
            action = packet["action"]
            xyz = np.array([action[f"delta_{axis}"] for axis in "xyz"], dtype=float)
            gripper = action["gripper"]
            if type(seq) is not int or seq <= self.last_seq:
                raise ValueError("输入序号必须递增")
            if type(enabled) is not bool:
                raise ValueError("enabled 必须是布尔值")
            if not np.isfinite(stamp) or not -0.05 <= now - stamp <= INPUT_TIMEOUT:
                raise ValueError("输入时间戳过期或超前")
            if not np.isfinite(xyz).all() or np.max(np.abs(xyz)) > 1.001:
                raise ValueError("XYZ 输入必须是 [-1, 1] 内的有限值")
            if gripper not in (0, 1, 2):
                raise ValueError("夹爪命令必须是官方 CLOSE=0/STAY=1/OPEN=2")
        except (KeyError, TypeError, ValueError) as error:
            self.pause("输入无效，已暂停")
            raise ValueError(str(error)) from error
        self.last_seq, self.last_stamp = seq, float(stamp)
        self.action, self.gripper = xyz, int(gripper)
        if not enabled:
            self.pause("RB 已松开，保持最后目标")
            self.can_enable = bool(np.max(np.abs(xyz)) == 0 and gripper == 1)
        elif self.can_enable:
            self.enabled = True
            self.reason = "手柄控制中"

    def _next_target(self):
        desired = self.action * TCP_SPEED / max(1.0, float(np.linalg.norm(self.action)))
        self.velocity += np.clip(
            desired - self.velocity,
            -TCP_ACCELERATION * self.dt,
            TCP_ACCELERATION * self.dt,
        )
        target = self.target.copy()
        if np.any(self.velocity):
            point = self.goal_xyz + self.velocity * self.dt
            if np.any(point < WORKSPACE_MIN) or np.any(point > WORKSPACE_MAX):
                self.pause("超出末端工作区")
                return self.target.copy()
            arm, error = self.ik.solve(point, self.target[:5])
            if error > 0.002 or not np.isfinite(arm).all():
                self.pause(f"IK 未收敛（{self.ik_mode}）：位置误差 {error * 1000:.2f} mm，上限 2 mm")
                return self.target.copy()
            if np.any(arm < self.ranges[:5, 0] - 1e-7) or np.any(arm > self.ranges[:5, 1] + 1e-7):
                self.pause(f"IK 关节解越界（{self.ik_mode}）")
                return self.target.copy()
            target[:5] += np.clip(arm - target[:5], -JOINT_SPEED * self.dt, JOINT_SPEED * self.dt)
        target[5] = np.clip(
            target[5] + (self.gripper - 1) * GRIPPER_SPEED * self.dt,
            max(-0.17, self.ranges[5, 0]),
            min(1.20, self.ranges[5, 1]),
        )
        if np.any(target < self.ranges[:, 0]) or np.any(target > self.ranges[:, 1]):
            self.pause("关节目标越界")
            return self.target.copy()
        self.preview.qpos[:] = self.data.qpos
        self.preview.qpos[self.qids] = target
        mujoco.mj_forward(self.model, self.preview)
        if self._collision(self.preview):
            self.pause("目标姿态碰撞")
            return self.target.copy()
        self.goal_xyz = grasp_point(self.model, self.preview, self.gripper_id)
        return target

    def advance(self, now):
        """推进一个原项目控制拍；超时后保持目标，禁止追赶积压输入。"""
        if self.last_stamp is None or now - self.last_stamp > INPUT_TIMEOUT:
            self.pause("输入超时，松开 RB 并回中后重新接管")
        previous = self.target.copy()
        if self.enabled:
            self.target = self._next_target()
        for index in range(SUBSTEPS):
            self.data.ctrl[self.aids] = previous + (self.target - previous) * ((index + 1) / SUBSTEPS)
            mujoco.mj_step(self.model, self.data)
            if self._collision(self.data):
                self.pause("物理仿真检测到碰撞")
                break
        if not np.isfinite(self.data.qpos).all():
            self.pause("仿真状态非有限值")
            raise RuntimeError(self.reason)
