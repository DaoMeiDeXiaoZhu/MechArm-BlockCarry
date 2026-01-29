from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

# ------------------------------------------------------------
# 导入所有模块（资产、MDP、事件、重置、终止）
# ------------------------------------------------------------

# 场景资产（机器人、物体、传感器）
from .assets_cfg import SceneAssetsCfg

# MDP：动作、观测、奖励、事件
from .mdp.actions_cfg import ActionsCfg
from .mdp.observations_cfg import ObservationsCfg
from .mdp.rewards_cfg import RewardsCfg
from .mdp.events_cfg import EventsCfg
from .mdp.terminations_cfg import TerminationsCfg


@configclass
class FirstRLEnvCfg(ManagerBasedRLEnvCfg):
    """
    统一注册所有环境模块的配置类。
    """
    # ------------------------------------------------------------
    # [关键修复] 1. 必须定义的仿真参数
    # ------------------------------------------------------------
    # 仿真步长倍率：Isaac Sim 默认物理步长是 60Hz (0.016s)
    # decimation=2 意味着策略网络的控制频率是 30Hz
    decimation = 6
    
    # 每个回合的最大时长 (秒)
    episode_length_s = 20

    # ------------------------------------------------------------
    # 2. 场景资产（机器人、物体、传感器）
    # ------------------------------------------------------------
    # [关键修复] 在这里必须指定环境数量和间距
    scene: SceneAssetsCfg = SceneAssetsCfg(num_envs=1024, env_spacing=2.5)

    # ------------------------------------------------------------
    # 3. 动作空间（MDP）
    # ------------------------------------------------------------
    actions: ActionsCfg = ActionsCfg()

    # ------------------------------------------------------------
    # 4. 观测空间（MDP）
    # ------------------------------------------------------------
    observations: ObservationsCfg = ObservationsCfg()

    # ------------------------------------------------------------
    # 5. 奖励系统（MDP）
    # ------------------------------------------------------------
    rewards: RewardsCfg = RewardsCfg()

    # ------------------------------------------------------------
    # 6. 事件系统（reset 时执行的逻辑）
    # ------------------------------------------------------------
    events: EventsCfg = EventsCfg()

    # ------------------------------------------------------------
    # 8. 终止条件（RL 层）
    # ------------------------------------------------------------
    terminations: TerminationsCfg = TerminationsCfg()