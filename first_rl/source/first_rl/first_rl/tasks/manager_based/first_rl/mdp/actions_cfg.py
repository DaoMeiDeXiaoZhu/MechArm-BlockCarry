# actions_cfg.py
# ================================================================
#  动作定义（Manager-Based 架构）
#  你可以在 env_cfg.py 中通过 ActionsCfg 引用本文件
# ================================================================

# 导入 Isaac Lab 的 configclass 装饰器，用于声明配置类，确保配置可序列化
from isaaclab.utils import configclass

# 导入 mdp 模块，其中包含 JointPositionActionCfg 等动作配置类型，用于定义 MDP 的动作空间
from isaaclab.envs import mdp


# 使用 configclass 声明这是一个配置类（Config Class）
@configclass
class ActionsCfg:
    """
    定义神经网络输出的动作类型及其映射逻辑。
    Manager-Based 架构中，动作由 ActionManager 自动处理：
    1. 每个定义的 ActionCfg 都会自动注册到 RL 算法的动作空间中。
    2. 算法输出的 [-1, 1] 范围内的值会被自动解包并应用到对应的关节。
    """

    # ------------------------------------------------------------
    # 1. 手臂关节控制：Joint Position Control (基于 PD 控制器)
    # ------------------------------------------------------------

    # 定义一个 RelativeJointPositionActionCfg，用于控制多个手臂关节的增量位置
    arm_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",      # 指定控制哪个资产（机器人），必须与 SceneAssetsCfg 中的属性名完全一致

        joint_names=[            # 指定要控制的关节名称列表（正则表达式或精确匹配）
            "shoulder_pan",      # 关节1：肩部基座水平旋转 (Yaw)
            "shoulder_lift",     # 关节2：肩部大臂抬升 (Pitch)
            "elbow_flex",        # 关节3：肘部弯曲 (Pitch)
            "wrist_flex",        # 关节4：手腕俯仰 (Pitch)
            "wrist_roll",        # 关节5：手腕侧向旋转 (Roll)
        ],

        scale=0.1,               # 动作缩放系数：
                                 # 计算公式：Target = Current + Action * Scale
                                 # 若网络输出 1.0，则实际目标关节角度在当前姿态基础上移动 0.1 rad
                                 # 较大的 scale 能让机器人动作更迅速，但过大会导致物理仿真不稳定
    )

    # ------------------------------------------------------------
    # 2. 夹爪控制：Joint Position Control (二值或连续控制)
    # ------------------------------------------------------------

    # 定义另一个 RelativeJointPositionActionCfg，专门用于控制末端执行器（夹爪）
    gripper_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",      # 同样控制 robot 资产，ActionManager 会自动合并所有关节指令

        joint_names=["gripper"], # 夹爪关节名，此处列表仅包含一个关节

        scale=0.08,               # 夹爪动作缩放：
                                 # 相比手臂，夹爪行程通常较短且需要更精细的控制，因此 scale 设得更小（0.2）
                                 # 网络输出 1.0 对应 0.08 rad 的位移
    )
    

# ================================================================
# 📌 机制级总结：
# 1. 自动维度推导：内置 PPO 算法会自动读取所有 joint_names 的总个数。
#    在本配置中，arm_pos(5个) + gripper_pos(1个) = 6维动作空间。
# 2. 控制频率：Action 会根据 env_cfg 中的 decimation 参数，在多个物理步长内持续施加控制力。
# 3. 限制保护：ActionManager 会自动将输出结果裁剪（Clip）在资产配置定义的关节限位（Limits）之内。
# ================================================================