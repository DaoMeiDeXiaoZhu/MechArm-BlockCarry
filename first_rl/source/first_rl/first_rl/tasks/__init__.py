import gymnasium as gym
from .manager_based.first_rl.first_rl_env_cfg import FirstRLEnvCfg
from .manager_based.first_rl.agents.rsl_rl_ppo_cfg import PPORunnerCfg

# 注册环境
gym.register(
    id="FirstRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 确保指向你的配置类 FirstRLEnvCfg (注意大小写)
        "env_cfg_entry_point": FirstRLEnvCfg,
        "rsl_rl_cfg_entry_point": PPORunnerCfg,
    },
)