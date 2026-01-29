from __future__ import annotations
import torch
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

def get_custom_scene_obs(env):
    """
    扁平化观测函数：直接从 robot body data 获取指尖坐标。
    同步奖励函数的获取逻辑，确保感知与反馈完全一致。
    """
    scene = env.scene
    robot = scene["robot"]
    cube = scene["cube"]
    env_origins = scene.env_origins
    
    # --- 1. 获取参考点刚体索引 (与奖励函数完全一致) ---
    # 假设你在奖励函数中使用的名字是 "finger1" 和 "finger2"
    link_names = ["finger1", "finger2"]
    link_indices, _ = robot.find_bodies(link_names)
    
    # --- 2. 提取世界坐标并转为环境局部坐标 ---
    tip1_env = robot.data.body_pos_w[:, link_indices[0], :] - env_origins
    tip2_env = robot.data.body_pos_w[:, link_indices[1], :] - env_origins
    cube_env = cube.data.root_pos_w - env_origins
    
    # --- 3. 计算 TCP (指尖中点) ---
    tcp_env = 0.5 * (tip1_env + tip2_env)
    
    # --- 4. 计算特征 ---
    # 指尖当前间距 (1维)
    finger_width = torch.norm(tip1_env - tip2_env, dim=-1, keepdim=True)
    
    # 物块相对 TCP 的位移 (3维)
    rel_cube_tcp = cube_env - tcp_env
    
    # --- 5. 组合最终观测向量 ---
    # 注意：建议这里也包含机械臂的基础位置数据，增强策略的全局感
    obs = [
        robot.data.joint_pos,                                # 关节位置 (例如 6或7维)
        torch.clamp(robot.data.joint_vel, -10.0, 10.0),      # 关节速度
        cube_env - tip1_env,                                 # 物块相对指尖1 (3维)
        cube_env - tip2_env,                                 # 物块相对指尖2 (3维)
        rel_cube_tcp,                                        # 物块相对TCP (3维)
        finger_width,                                        # 两指尖距离 (1维)
        cube_env[:, 1:2],                                    # 物块环境 Y (1维) - 用于导航目标点
        cube_env[:, 2:3]                                     # 物块环境 Z (1维) - 用于判断是否提起
    ]
    
    return torch.cat(obs, dim=-1)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 这里的名字可以保持不变，但内部逻辑已经更新
        full_scene = ObsTerm(func=get_custom_scene_obs)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()