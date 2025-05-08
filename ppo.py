import retro
import numpy as np
import torch
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env_created = False

def make_env():
    global env_created
    if env_created:
        raise ValueError("Environment already created. Close it first.")
    env_created = True
    return retro.make(game="SuperMarioBros-Nes", state="Level1-1")

total_timesteps = 10_000_000
save_every = 10_000

os.makedirs("ppo_output", exist_ok=True)
rewards_file = open("ppo_output/rewards.txt", "w")
averages_file = open("ppo_output/averages.txt", "w")

rewards_buffer = []
average_buffer = []

env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")

vec_env = DummyVecEnv([lambda: env])

model = PPO("CnnPolicy", vec_env, verbose=1, device="auto")

n_steps = 0
start_time = time.time()

try:
    while n_steps < total_timesteps:
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        
        obs = vec_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
        
        rewards_buffer.append(total_reward)
        
        # Calculate moving average (last 10 episodes or all if less than 10)
        moving_avg = np.mean(rewards_buffer[-10:])
        average_buffer.append(moving_avg)

        rewards_file.write(f"{total_reward}\n")
        rewards_file.flush()
        averages_file.write(f"{moving_avg}\n")
        averages_file.flush()

        n_steps += save_every
        
        elapsed_time = time.time() - start_time
        
        print(f"Step: {n_steps}/{total_timesteps} | Reward: {total_reward:.2f} | Moving Avg: {moving_avg:.2f} | Time: {elapsed_time:.1f}s")

        if n_steps % 100000 == 0:
            model.save(f"ppo_output/mario_model_{n_steps}")

    model.save("ppo_output/mario_model_final")
    
    print("Training complete!")
    print(f"Final average reward (last 10 episodes): {moving_avg:.2f}")

finally:
    rewards_file.close()
    averages_file.close()
    vec_env.close()
