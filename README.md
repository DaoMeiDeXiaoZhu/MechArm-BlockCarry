**项目说明：**

该项目是使用机械臂夹取物块的任务，已经从实验效果方面展示了模型的能力，逻辑上没有问题，可设计对应的奖励函数来实现更复杂的任务

**项目结构：**

<img width="3419" height="1374" alt="yuque_diagram (3)" src="https://github.com/user-attachments/assets/311a1e1e-5400-42ab-86d2-d974634d3789" />

**文件说明：**

1. task/__init__.py：将我们所写的强化学习配置文件first_rl_env_cfg.py与强化学习算法rsl_rl_ppo_cfg.py绑定在一起形成任务。
2. first_rl_env_cfg.py：这里面会定义仿真步长与回合持续时间，并且将马尔可夫决策mdp过程中写的配置文件进行统一注册。
3. angents/rsl_rl_ppo_cfg.py：这里存放的是rsl_rl框架下自带的 ppo 算法，可以修改里面的参数来调整算法收敛速度与稳定性等。
4. mdp/observations_cfg.py：观测空间，定义了机械臂能够观测到的数据。
5. mdp/actions_cfg.py：动作空间，定义了机械臂各个可活动关节的运动幅度。
6. mdp/rewards_cfg.py：奖励函数，定义了机械臂在当前观测空间执行动作后进入到下一观测空间后获得的奖励大小。
7. mdp/terminations_cfg.py：终止逻辑，包含时间步终止和任务失败与成功的终止。
8. mdp/events_cfg.py：当mdp/terminations_cfg.py返回值为 True 时表示环境要重置，此时需要执行该文件中定义的逻辑进行环境重置。

**奖励曲线：**

