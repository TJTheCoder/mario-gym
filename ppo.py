import os
import json
import gymnasium as gym
import retro
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gym_retro.retro_wrapper import RetroArcade

LOG_DIR = "./ppo_logs/Mario/"
REWARD_TXT_PATH = os.path.join(LOG_DIR, "ppo_rewards.txt")
GRAPH_IMG_PATH = os.path.join(LOG_DIR, "ppo_reward_curve.png")

os.makedirs(LOG_DIR, exist_ok=True)

GAME_START_DICT = {
    "SuperMarioBros-Nes": ["Level1-1"]
}

# Callback to save rewards and update graph
class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.rewards.append(reward)
                    with open(REWARD_TXT_PATH, "a") as f:
                        f.write(f"{reward}\n")
                    self._plot_rewards()
        return True

    def _plot_rewards(self):
        if len(self.rewards) < 10:
            return
        smoothed = np.convolve(self.rewards, np.ones(50)/50, mode="valid")
        x_vals = np.arange(len(smoothed))

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward")
        plt.title("PPO Training Curve - SuperMarioBros-Nes Level1-1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAPH_IMG_PATH)
        plt.close()

if __name__ == "__main__":
    env = RetroArcade(game_start_dict=GAME_START_DICT, use_discrete_actions=True)
    env = Monitor(env, LOG_DIR)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    callback = RewardLoggerCallback(check_freq=1000)

    model.learn(total_timesteps=10_000_000, callback=callback)

    env.close()
