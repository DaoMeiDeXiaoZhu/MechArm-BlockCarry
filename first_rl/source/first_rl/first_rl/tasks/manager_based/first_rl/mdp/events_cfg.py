# ================================================================
#  events_cfg.py
#  Isaac Lab 事件与重置配置 - 强化学习任务专用
# ================================================================

from __future__ import annotations

import torch
import numpy as np
from typing import List
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

##
# 自定义事件函数 (Custom Event Functions)
##

def reset_cube_to_left_table(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    cube_name: str = "cube"
):
    """
    📌 自定义重置逻辑：物块强制左侧分布 (y > 0.3)
    ------------------------------------------------
    该函数在环境重置时调用，确保物块出现在机器人视角的左侧区域。
    """
    num_envs = len(env_ids)
    device = env.device
    asset: RigidObject = env.scene[cube_name]
    
    # 获取当前需要重置的环境在世界空间的原点偏移
    env_origins = env.scene.env_origins[env_ids]
    
    # --- 采样范围定义 ---
    # X 轴：对应桌子的深度方向 (0.6 到 0.9 是安全抓取深度)
    x_range = (-0.3, 0.3)   
    
    # Y 轴：根据你的要求，强制设定在 0.3 以上
    # 假设桌面边缘在 0.5 左右，采样区间为 [0.3, 0.5]
    y_range = (0.2, 0.5)    
    
    # Z 轴：桌面高度 (0.5) + 物块半高 (0.025) + 缓冲 (0.001)
    z_fixed = 0.526          
    
    # 1. 在指定范围内进行均匀随机采样
    rand_x = torch.rand(num_envs, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    rand_y = torch.rand(num_envs, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    rand_z = torch.full((num_envs,), z_fixed, device=device)
    
    # 合成环境局部坐标
    local_pos = torch.stack([rand_x, rand_y, rand_z], dim=-1) 
    
    # 2. 构建 Root States
    # 克隆资产默认的根节点状态（包含 Scale 等信息）
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # 将采样得到的局部坐标转换为世界坐标注入 root_states
    root_states[:, 0:3] = env_origins + local_pos
    
    # 3. 随机偏航角 (Yaw Rotation)
    # 让物块在桌面上随机转动角度，增加抓取难度
    rand_yaw = torch.rand(num_envs, device=device) * 2 * np.pi
    root_states[:, 3] = torch.cos(rand_yaw / 2.0) # qw
    root_states[:, 6] = torch.sin(rand_yaw / 2.0) # qz
    
    # 4. 动力学清零
    # 重置瞬间必须清除速度 (linear + angular)，防止物体继承上个回合的动量飞出去
    root_states[:, 7:13] = 0.0

    # 5. 写入物理引擎
    asset.write_root_state_to_sim(root_states, env_ids)


@configclass
class EventsCfg:
    """
    📌 事件管理配置类
    """
    
    # 机制：重置时将机器人恢复至初始姿态
    reset_robot = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # 机制：执行上述自定义的“左侧采样”逻辑
    reset_cube = EventTerm(
        func=reset_cube_to_left_table,
        mode="reset",
        params={
            "cube_name": "cube"  
        }
    )

    # 📌 域随机化 (Domain Randomization)：
    # 在重置时为各个关节添加 ±0.05 rad 的位置噪声，防止模型过拟合
    reset_robot_joints_sample = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )