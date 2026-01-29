# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # 机器人任务通常 24-32 即可，加快迭代
    max_iterations = 5000   # 搬运任务比推杆难，需要更多迭代
    save_interval = 50
    experiment_name = "cube_transport_task"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 增加网络深度。搬运任务涉及坐标变换，64x64 太浅了
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 提高 Critic 权重，让它更准确地估计高度/距离价值
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,   # 略微降低熵系数，因为 4096 并行的探索量已经足够大
        num_learning_epochs=5,
        # 【关键修改】大幅增加 mini_batches。
        # 4096 envs * 32 steps / 32 mini_batches = 4096 per batch，这是一个很健康的数值
        num_mini_batches=32, 
        learning_rate=5e-4,   # 4096 并行可以稍微调大一点点学习率
        schedule="adaptive",
        gamma=0.98,           # 搬运任务更看重中短期奖励，降低一点 gamma
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,    # 稍微放宽梯度裁剪，允许更有力的更新
    )