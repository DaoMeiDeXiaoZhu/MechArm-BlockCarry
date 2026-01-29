from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- 全局记忆库：用于跨帧存储 ---
_LAST_GRIPPER_ACTION = None  # 记录上一步的动作（用于 MDP 正确性）
_LAST_FINGER_DIST = None     # 上一步爪子间距
_HAS_BEEN_LIFTED = None      # 记录物块是否被提起过


def cube_transport_linear_reward(
    env: ManagerBasedRLEnv,
    cube_name: str = "cube",
    robot_name: str = "robot",
    finger1_name: str = "finger1",
    finger2_name: str = "finger2",
    table_height: float = 0.5,
    cube_size: float = 0.05,
    max_ee_cube_dist: float = 1.2,
    target_lift_height: float = 0.2,
    max_y_dist: float = 1.2,
) -> torch.Tensor:

    global _LAST_GRIPPER_ACTION, _LAST_FINGER_DIST, _HAS_BEEN_LIFTED

    # --- 1. 数据准备 ---
    scene = env.scene
    cube = scene[cube_name]
    robot = scene[robot_name]
    env_origins = scene.env_origins
    num_envs = env.num_envs

    # 获取指尖位置与间距
    link_indices, _ = robot.find_bodies([finger1_name, finger2_name])
    curr_tip1 = robot.data.body_pos_w[:, link_indices[0], :] - env_origins
    curr_tip2 = robot.data.body_pos_w[:, link_indices[1], :] - env_origins
    curr_finger_dist = torch.norm(curr_tip1 - curr_tip2, dim=-1)

    # 夹爪物理极限
    f_min, f_max = 0.0080949645, 0.2580147982

    # --- 2. 初始化记忆库 ---
    if _LAST_GRIPPER_ACTION is None or _LAST_GRIPPER_ACTION.shape[0] != num_envs:
        _LAST_GRIPPER_ACTION = torch.zeros(num_envs, device=env.device)

    if _LAST_FINGER_DIST is None or _LAST_FINGER_DIST.shape[0] != num_envs:
        _LAST_FINGER_DIST = curr_finger_dist.clone()

    if _HAS_BEEN_LIFTED is None or _HAS_BEEN_LIFTED.shape[0] != num_envs:
        _HAS_BEEN_LIFTED = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    # 检测环境是否重置
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
        _HAS_BEEN_LIFTED[reset_env_ids] = False

    # --- 3. 基础判定 ---
    current_gripper_action = env.action_manager.action[:, -1]  # 当前动作 a_t
    prev_gripper_action = _LAST_GRIPPER_ACTION                 # 上一帧动作 a_{t-1}

    cube_env = cube.data.root_pos_w - env_origins
    cube_height = cube_env[:, 2] - table_height - (cube_size / 2.0)
    tcp_env = 0.5 * (curr_tip1 + curr_tip2)
    dist_ee_to_cube = torch.norm(cube_env - tcp_env, dim=-1)

    # --- ★★★ 关键修改：夹紧判定必须基于上一帧动作 ★★★
    dist_diff = torch.abs(curr_finger_dist - _LAST_FINGER_DIST)
    is_static = dist_diff < 1e-4
    is_clamped = (prev_gripper_action != 0) & is_static & (0.03 < curr_finger_dist) & (curr_finger_dist < 0.1)

    # 更新提起记录
    _HAS_BEEN_LIFTED = _HAS_BEEN_LIFTED | (is_clamped & (cube_height > 0.03))

    # --- 4. 奖励计算逻辑 ---
    total_reward = torch.zeros(num_envs, device=env.device)

    # 夹紧固定奖励
    clamp_bonus = 1.0
    total_reward += is_clamped.float() * clamp_bonus

    # [规则 1] 距离物块越近奖励越高
    approach_reward = torch.clamp((max_ee_cube_dist - dist_ee_to_cube) / max_ee_cube_dist, min=0.0)
    total_reward += approach_reward * 1.0

    # [规则 2&3] 夹爪姿态引导 (远张近合)
    near_mask = (dist_ee_to_cube <= 0.015)
    pose_reward = torch.zeros(num_envs, device=env.device)

    # --- 1. 针对“远”的情况 (near_mask 没生效)：目标 0.1，两端归零 ---
    target_far_dist = 0.1
    # 左侧斜率：f_min -> 0.1 (0.0 -> 1.0)
    far_reward_left = (curr_finger_dist - f_min) / (target_far_dist - f_min + 1e-6)
    # 右侧斜率：0.1 -> f_max (1.0 -> 0.0)
    far_reward_right = (f_max - curr_finger_dist) / (f_max - target_far_dist + 1e-6)
    
    # 结合成三角形函数
    far_pose_reward = torch.where(curr_finger_dist < target_far_dist, far_reward_left, far_reward_right)

    # --- 2. 针对“近”的情况 (near_mask 生效)：目标 0.04 (假设为 0.04m)，闭合引导 ---
    # 你提到的 0.4 超过了 f_max(0.258)，在 IsaacLab 常见单位中通常指 0.04m (4cm)
    target_near_dist = 0.04 
    # 这里使用线性函数：当开度从 f_max 减小到 target_near_dist 时，奖励从 0 升至 1
    near_pose_reward = (f_max - curr_finger_dist) / (f_max - target_near_dist + 1e-6)

    # --- 3. 汇总与约束 ---
    # 统一进行 clamp 保证奖励在 [0, 1] 区间，防止超出物理极限导致的负值
    pose_reward[~near_mask] = torch.clamp(far_pose_reward[~near_mask], min=0.0, max=1.0) * 0.2
    pose_reward[near_mask] = torch.clamp(near_pose_reward[near_mask], min=0.0, max=1.0)
    
    total_reward += pose_reward * 1.0

    # [规则 4&5] 提升与运输奖励 (仅在夹紧时)
    is_clamped_float = is_clamped.float()
    
    # 定义是否进入降落区 (y < -0.3)
    is_in_drop_zone = (cube_env[:, 1] < -0.3)

    # 计算高度偏差 (目标 0.1)
    lift_error = torch.abs(cube_height - target_lift_height)
    lift_reward = torch.exp(-20.0 * lift_error)
    lift_reward = is_clamped_float * (~is_in_drop_zone).float() * lift_reward * 2.0

    # --- 关键修改：只有不在降落区时，才给提升奖励 ---
    total_reward += lift_reward

    # --- 运输奖励逻辑 ---
    target_y = -0.35
    dist_to_y_goal = torch.abs(cube_env[:, 1] - target_y)
    transport_reward = torch.clamp((max_y_dist - dist_to_y_goal) / max_y_dist, min=0.0)

    # 运输奖励触发条件：夹紧、且高度在目标高度附近（比如偏差小于 0.05m）
    at_lift_height = (lift_error < 0.1).float()
    transport_reward = is_clamped_float * at_lift_height * transport_reward * 4.0
    total_reward += transport_reward

    # [规则 6] 降落引导 (仅在进入降落区且夹紧时)
    # 目标：高度从 target_lift_height 降到 0
    descend_reward = torch.exp(-10.0 * torch.clamp(cube_height, min=0.0))
    descend_reward = is_clamped_float * is_in_drop_zone.float() * descend_reward * 2.0

    # --- 关键修改：使用 is_in_drop_zone 作为开关 ---
    total_reward += descend_reward

    # --- 5. 成功与失败判定 (简化版) ---
    target_y = -0.35
    dist_to_y_goal = torch.abs(cube_env[:, 1] - target_y)
    
    # 定义成功条件：物块在目标点附近 (0.05m) 且 高度在桌面上 (0.05m以内)
    # 不再判断是否松手，只要带着物块到这里就算赢
    is_at_goal_pos = (dist_to_y_goal < 0.05) & (cube_height < 0.05)
    
    # 任务成功判定：只要到达目标位置就算成功
    is_success = is_at_goal_pos & _HAS_BEEN_LIFTED

    # 失败判定：保持原来的掉落判定（如果还没到终点就松手了）
    is_in_drop_zone = (cube_env[:, 1] < -0.3)
    dropped_midway = _HAS_BEEN_LIFTED & (~is_clamped) & (~is_at_goal_pos)
    
    out_of_table = (cube_env[:, 0].abs() > 0.4) | (cube_env[:, 1] > 0.6) | (cube_height < -0.05)

    # --- 6. 应用大奖与惩罚 ---
    success_reward = 30.0
    # 只要满足 success，这一帧就给大奖
    total_reward[is_success] += success_reward

    # 应用惩罚
    total_reward[dropped_midway] -= 5.0
    total_reward[out_of_table] -= 10.0

    # --- 更新跨步状态（必须在最后） ---
    _LAST_GRIPPER_ACTION = current_gripper_action.clone()
    _LAST_FINGER_DIST = curr_finger_dist.clone()
    
    # print('\n================================')
    # print(f'夹爪间距={curr_finger_dist}')
    # if is_success.any():
    #     print('==================任务成功======================')
    # if dropped_midway.any():
    #     print('==================中途松手======================')
    # print(f'靠近奖励={approach_reward.item()}\n夹爪开合奖励={pose_reward.item()}\n提升奖励={lift_reward.item()}\n运输奖励={transport_reward.item()}\n降落引导={descend_reward.item()}\n物块掉落惩罚={dropped_midway.float()*-2.0}')

    # 步数惩罚
    total_reward -= 0.1

    # print(f'夹爪与物块距离={dist_ee_to_cube}，是否夹紧={is_clamped}，与目标y的距离={dist_to_y_goal}，\
    #       物块高度={cube_height}，是否掉出桌面={out_of_table}，夹紧后是否松手={dropped_midway}，是否被提起过={_HAS_BEEN_LIFTED}')

    return total_reward


@configclass
class RewardsCfg:
    transport_task = RewardTermCfg(
        func=cube_transport_linear_reward,
        weight=1.0,
        params={
            "cube_name": "cube",
            "robot_name": "robot",
            "finger1_name": "finger1",
            "finger2_name": "finger2",
            "cube_size": 0.05,
            "table_height": 0.5,
            "max_ee_cube_dist": 1.0,
            "target_lift_height": 0.1,
            "max_y_dist": 0.8,
        }
    )

    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01
    )
