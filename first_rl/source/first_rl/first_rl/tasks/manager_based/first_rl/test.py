import os
import sys
import torch
from pynput import keyboard
from isaaclab.app import AppLauncher

# 1. 启动仿真引擎
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 注意：这里改为导入 ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnv

# 2. 路径补丁
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import first_rl.first_rl_env_cfg as env_module
    FirstRLEnvCfg = env_module.FirstRLEnvCfg
    print("✅ 配置加载成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    simulation_app.close()
    sys.exit()


class TeleopController:
    def __init__(self, action_dim, device):
        self.device = device
        self.actions = torch.zeros((1, action_dim), device=device)

        max_idx = min(action_dim, 6)
        self.key_map = {}
        base_pairs = [
            ('1', 'q'), ('2', 'w'), ('3', 'e'),
            ('4', 'r'), ('5', 't'), ('6', 'y'),
        ]
        for i in range(max_idx):
            pos_key, neg_key = base_pairs[i]
            self.key_map[pos_key] = (i, +1.0)
            self.key_map[neg_key] = (i, -1.0)

        print(f"🎮 TeleopController 初始化完成，action_dim = {action_dim}")

    def on_press(self, key):
        try:
            char = getattr(key, 'char', None)
            if char in self.key_map:
                idx, val = self.key_map[char]
                self.actions[0, idx] = val
        except: pass

    def on_release(self, key):
        try:
            char = getattr(key, 'char', None)
            if char in self.key_map:
                idx, _ = self.key_map[char]
                self.actions[0, idx] = 0.0
        except: pass


def main():
    cfg = FirstRLEnvCfg()
    cfg.scene.num_envs = 1
    
    # --- 核心修改点 1: 使用 ManagerBasedRLEnv ---
    # 只有 RLEnv 会根据 cfg 中的 RewardsCfg 自动初始化 RewardManager
    env = ManagerBasedRLEnv(cfg=cfg)

    action_shape = env.action_manager.action.shape
    action_dim = action_shape[1]
    device = env.device
    print(f"✅ 环境动作维度: {action_dim}, device = {device}")

    controller = TeleopController(action_dim=action_dim, device=device)

    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()

    print("\n" + "=" * 60)
    print("🚀 实时示教 + 奖励监控模式已开启")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = controller.actions
            
            # --- 核心修改点 2: 解构 RL 环境的五个返回值 ---
            # obs: 观测, rew: 奖励, terminated: 终止, truncated: 超时, extras: 额外信息
            obs, rew, terminated, truncated, extras = env.step(actions)

            # --- 核心修改点 3: 实时打印奖励 ---
            # rew 的 shape 是 (num_envs,)，我们只有 1 个环境，所以取 [0]
            current_rew = rew[0].item()
            
            # 使用 sys.stdout.write 实现单行刷新，避免刷屏
            sys.stdout.write(f"\r当前奖励值: {current_rew:10.4f} | 终止状态: {terminated[0]}")
            sys.stdout.flush()

            # 如果触发了终止条件（比如掉下平台或完成任务），重置环境
            if terminated[0] or truncated[0]:
                print("\n🔄 检测到环境重置...")
                env.reset()

    simulation_app.close()


if __name__ == "__main__":
    main()