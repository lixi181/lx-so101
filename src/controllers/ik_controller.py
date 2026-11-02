"""SO-101 末端逆运动学（IK）求解器与配套运动学工具。

阻尼最小二乘（DLS）IK：3 个位置约束 + 末端朝下约束 + 可选夹爪 yaw 约束。
求解过程中临时改写 qpos/qvel，函数返回前完整恢复调用方的物理状态，
因此可以在任意时刻对一个"干净"的仿真状态查询目标关节角而不产生副作用。
"""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.square_grasp_planner import wrap_to_pi

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRASP_OFFSET = np.array([0.010, 0.0, -0.080])
DOWN = np.array([0.0, 0.0, -1.0])


def arm_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    qpos_ids = []
    dof_ids = []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"找不到关节: {name}")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
    return qpos_ids, dof_ids


def grasp_point(model: mujoco.MjModel, data: mujoco.MjData, gripper_id: int) -> np.ndarray:
    rotation = np.asarray(data.xmat[gripper_id]).reshape(3, 3)
    return np.asarray(data.xpos[gripper_id]) + rotation @ GRASP_OFFSET


def gripper_closing_axis(model: mujoco.MjModel, data: mujoco.MjData, gripper_id: int) -> np.ndarray:
    """夹爪闭合方向（世界坐标系）。

    实测：moving_jaw_tip 从张开到闭合的位移在 gripper 局部坐标中为
    (-0.0797, 0.0000, -0.0390)，Y 分量为 0，即夹爪沿自身 x 轴开合、绕 y 轴转动。
    因此闭合方向取旋转矩阵第一列。
    """
    rotation = np.asarray(data.xmat[gripper_id]).reshape(3, 3)
    return rotation[:, 0].copy()


def gripper_yaw(model: mujoco.MjModel, data: mujoco.MjData, gripper_id: int) -> float:
    """夹爪闭合方向在世界 XY 平面内的 yaw（俯视姿态下即夹爪朝向）。"""
    closing = gripper_closing_axis(model, data, gripper_id)
    return float(np.arctan2(closing[1], closing[0]))


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    qpos_ids: list[int],
    dof_ids: list[int],
    gripper_id: int,
    target_yaw: float | None = None,
) -> tuple[np.ndarray, float]:
    """用位置、末端朝向和可选 yaw 的阻尼最小二乘 IK，保留当前物理状态。

    `target_yaw` 是夹爪闭合方向在世界 XY 平面内的目标 yaw。俯视抓取需要 3 个位置
    约束加 1 个朝下约束，剩下 wrist_roll 用于控制 yaw；额外的 yaw 约束写成第 7 行，
    用世界 z 轴角速度雅可比驱动，超定部分由阻尼最小二乘折中。
    """
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    q = data.qpos[qpos_ids].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(200):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        point = grasp_point(model, data, gripper_id)
        rotation = np.asarray(data.xmat[gripper_id]).reshape(3, 3)
        approach = -rotation[:, 2]
        error = [target - point, np.cross(approach, DOWN)]
        if target_yaw is not None:
            closing = rotation[:, 0]
            current_yaw = np.arctan2(closing[1], closing[0])
            error.append([wrap_to_pi(target_yaw - current_yaw)])
        error_array = np.concatenate(error)
        if np.linalg.norm(error_array) < 2e-4:
            break
        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jac(model, data, jacp, jacr, point, gripper_id)
        rows = [jacp[:, dof_ids], jacr[:, dof_ids]]
        if target_yaw is not None:
            # 绕世界 z 轴的转动分量：驱动夹爪闭合方向的 yaw 误差。
            rows.append(jacr[2:3, dof_ids])
        jac = np.vstack(rows)
        damping = 1e-3
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(jac.shape[0]), error_array)
        q += 0.65 * delta
        for index, qpos_id in enumerate(qpos_ids):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[index])
            q[index] = np.clip(q[index], model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])

    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)
    final_error = np.linalg.norm(target - grasp_point(model, data, gripper_id))
    data.qpos[:] = original_qpos
    data.qvel[:] = original_qvel
    mujoco.mj_forward(model, data)
    return q, float(final_error)
