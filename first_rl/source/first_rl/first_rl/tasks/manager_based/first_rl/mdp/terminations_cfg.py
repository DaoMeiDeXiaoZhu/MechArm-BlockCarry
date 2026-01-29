from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as Term
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- 全局记忆库：用于跨帧存储 ---
_LAST_GRIPPER_ACTION = None      # 上一帧动作
_LAST_FINGER_DIST = None         # 上一帧指距
_HAS_BEEN_LIFTED = None          # 是否曾经提起过
_IS_CLAMP = None                 # 当前是否夹紧


def _compute_state(
    env: ManagerBasedRLEnv,
    cube_name: str = "cube",
    robot_name: str = "robot",
    finger1_name: str = "finger1",
    finger2_name: str = "finger2",
    table_height: float = 0.5,
    cube_size: float = 0.05,
):
    global _LAST_GRIPPER_ACTION, _LAST_FINGER_DIST, _HAS_BEEN_LIFTED, _IS_CLAMP

    scene = env.scene
    cube = scene[cube_name]
    robot = scene[robot_name]
    env_origins = scene.env_origins
    num_envs = env.num_envs

    # 获取指尖位置
    link_indices, _ = robot.find_bodies([finger1_name, finger2_name])
    curr_tip1 = robot.data.body_pos_w[:, link_indices[0], :] - env_origins
    curr_tip2 = robot.data.body_pos_w[:, link_indices[1], :] - env_origins
    curr_finger_dist = torch.norm(curr_tip1 - curr_tip2, dim=-1)

    # 1. 初始化逻辑：完全按照你的方式，确保不出现 None 下标报错
    if _LAST_GRIPPER_ACTION is None or _LAST_GRIPPER_ACTION.shape[0] != num_envs:
        _LAST_GRIPPER_ACTION = torch.zeros(num_envs, device=env.device)

    if _LAST_FINGER_DIST is None or _LAST_FINGER_DIST.shape[0] != num_envs:
        _LAST_FINGER_DIST = curr_finger_dist.clone()

    if _HAS_BEEN_LIFTED is None or _HAS_BEEN_LIFTED.shape[0] != num_envs:
        _HAS_BEEN_LIFTED = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    if _IS_CLAMP is None or _IS_CLAMP.shape[0] != num_envs:
        _IS_CLAMP = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    # 2. Reset 逻辑
    reset_mask = (env.episode_length_buf <= 1)
    if reset_mask.any():
        _HAS_BEEN_LIFTED[reset_mask] = False
        _IS_CLAMP[reset_mask] = False
        _LAST_GRIPPER_ACTION[reset_mask] = 0.0
        _LAST_FINGER_DIST[reset_mask] = curr_finger_dist[reset_mask]

    # 3. 判定更新
    current_gripper_action = env.action_manager.action[:, -1]
    prev_gripper_action = _LAST_GRIPPER_ACTION                 

    cube_env = cube.data.root_pos_w - env_origins
    cube_height = cube_env[:, 2] - table_height - cube_size / 2.0

    dist_diff = torch.abs(curr_finger_dist - _LAST_FINGER_DIST)
    is_static = dist_diff < 1e-4

    # ★★★ 关键点：夹紧判定增加 0.1 限制（与 reward 保持一致）
    _IS_CLAMP = (
        (prev_gripper_action != 0)
        & is_static
        & (curr_finger_dist > 0.03)
        & (curr_finger_dist < 0.1)
    )

    # lifted 判定
    _HAS_BEEN_LIFTED |= (_IS_CLAMP & (cube_height > 0.03))

    # 更新跨步状态
    _LAST_GRIPPER_ACTION = current_gripper_action.clone()
    _LAST_FINGER_DIST = curr_finger_dist.clone()


def _safe_lifted_and_clamp(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    global _HAS_BEEN_LIFTED, _IS_CLAMP
    # 这里的 None 检查是为了防止初始化第一帧调用的异常
    if (_HAS_BEEN_LIFTED is None) or (_IS_CLAMP is None):
        return torch.zeros_like(env_ids, dtype=torch.bool), torch.zeros_like(env_ids, dtype=torch.bool)
    return _HAS_BEEN_LIFTED[env_ids], _IS_CLAMP[env_ids]


def task_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    cube_name: str = "cube",
    finger1: str = "finger1",
    finger2: str = "finger2",
    table_height: float = 0.5,
    cube_size: float = 0.05,
) -> torch.Tensor:
    _compute_state(env, cube_name, "robot", finger1, finger2, table_height, cube_size)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    cube = env.scene[cube_name]
    env_origins = env.scene.env_origins
    cube_pos = (cube.data.root_pos_w - env_origins)[env_ids]
    cube_height = cube_pos[:, 2] - table_height - cube_size / 2.0

    target_y = -0.35
    dist_to_y_goal = torch.abs(cube_pos[:, 1] - target_y)
    is_at_goal = (dist_to_y_goal < 0.05) & (cube_height < 0.05)
    
    lifted, clamped = _safe_lifted_and_clamp(env, env_ids)

    # 计算成功结果
    success_mask = lifted & is_at_goal

    # # --- 打印调试信息 ---
    # if success_mask.any():
    #     success_env_ids = env_ids[success_mask].tolist()
    #     print(f"\033[92m[TERMINATION: SUCCESS]\033[0m 环境 {success_env_ids} 满足成功重置条件！")

    return success_mask


def task_fail_drop(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    cube_name: str = "cube",
    finger1: str = "finger1",
    finger2: str = "finger2",
    table_height: float = 0.5,
    cube_size: float = 0.05,
) -> torch.Tensor:
    # 状态更新已在 success 中跑过，此处直接读取
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    cube = env.scene[cube_name]
    env_origins = env.scene.env_origins
    cube_pos = (cube.data.root_pos_w - env_origins)[env_ids]
    cube_height = cube_pos[:, 2] - table_height - cube_size / 2.0
    
    target_y = -0.35
    dist_to_y_goal = torch.abs(cube_pos[:, 1] - target_y)
    is_at_goal = (dist_to_y_goal < 0.05) & (cube_height < 0.05)
    
    lifted, clamped = _safe_lifted_and_clamp(env, env_ids)

    # 计算失败结果
    fail_mask = lifted & (~clamped) & (~is_at_goal)

    # # --- 打印调试信息 ---
    # if fail_mask.any():
    #     fail_env_ids = env_ids[fail_mask].tolist()
    #     print(f"\033[91m[TERMINATION: FAIL_DROP]\033[0m 环境 {fail_env_ids} 掉落重置！(曾经提起但未在终点松手)")

    return fail_mask


def cube_out_of_table(env, env_ids=None, cube_name="cube", table_height=0.5, cube_size=0.05):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    cube = env.scene[cube_name]
    env_origins = env.scene.env_origins
    cube_env = cube.data.root_pos_w - env_origins
    pos = cube_env[env_ids]
    cube_height = pos[:, 2] - table_height - (cube_size / 2.0)
    
    out_mask = (pos[:, 0].abs() > 0.4) | (pos[:, 1] > 0.6) | (cube_height < -0.1)

    # # --- 打印调试信息 ---
    # if out_mask.any():
    #     out_env_ids = env_ids[out_mask].tolist()
    #     print(f"\033[93m[TERMINATION: OUT_OF_TABLE]\033[0m 环境 {out_env_ids} 物块越界/掉下桌子！")

    return out_mask


@configclass
class TerminationsCfg:
    time_out = Term(func=mdp.time_out, time_out=True)

    success = Term(
        func=task_success,
        params={"cube_name": "cube", "finger1": "finger1", "finger2": "finger2"},
    )

    fail_drop = Term(
        func=task_fail_drop,
        params={"cube_name": "cube", "finger1": "finger1", "finger2": "finger2"},
    )

    cube_out = Term(
        func=cube_out_of_table,
        params={"cube_name": "cube"},
    )