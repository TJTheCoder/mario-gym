import retro
import numpy as np
import torch
import os
import time
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

total_timesteps = 10_000_000
save_every = 10_000
starting_step = 3_000_000

os.makedirs("ppo_output", exist_ok=True)

if os.path.exists("ppo_output/rewards.txt"):
    shutil.copy("ppo_output/rewards.txt", "ppo_output/rewards2.txt")
else:
    open("ppo_output/rewards2.txt", "w").close()

if os.path.exists("ppo_output/averages.txt"):
    shutil.copy("ppo_output/averages.txt", "ppo_output/averages2.txt")
else:
    open("ppo_output/averages2.txt", "w").close()

rewards_file = open("ppo_output/rewards2.txt", "a")
averages_file = open("ppo_output/averages2.txt", "a")

# Load existing reward history
rewards_buffer = []
try:
    with open("ppo_output/rewards2.txt", "r") as f:
        for line in f:
            if line.strip():
                rewards_buffer.append(float(line.strip()))
    print(f"Loaded {len(rewards_buffer)} reward entries from previous training")
except Exception as e:
    print(f"Could not load previous rewards: {e}")
    rewards_buffer = []

average_buffer = []
try:
    with open("ppo_output/averages2.txt", "r") as f:
        for line in f:
            if line.strip():
                average_buffer.append(float(line.strip()))
    print(f"Loaded {len(average_buffer)} average entries from previous training")
except Exception as e:
    print(f"Could not load previous averages: {e}")
    average_buffer = []

env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")

vec_env = DummyVecEnv([lambda: env])

print(f"Loading existing model from ppo_output/mario_model_{starting_step}.zip")
model = PPO.load(
    f"ppo_output/mario_model_{starting_step}", 
    env=vec_env,
    device="auto"
)

n_steps = starting_step
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
        
        steps_done = n_steps - starting_step
        steps_remaining = total_timesteps - n_steps
        time_per_step = elapsed_time / steps_done if steps_done > 0 else 0
        estimated_time_remaining = time_per_step * steps_remaining if time_per_step > 0 else 0
        
        print(f"Step: {n_steps}/{total_timesteps} | Reward: {total_reward:.2f} | Moving Avg: {moving_avg:.2f} | Time: {elapsed_time:.1f}s | Est. Remaining: {estimated_time_remaining/3600:.1f}h")

        if n_steps % 100000 == 0:
            model.save(f"ppo_output/mario_model_{n_steps}")

    model.save("ppo_output/mario_model_final")
    
    print("Training complete!")
    print(f"Final average reward (last 10 episodes): {moving_avg:.2f}")

finally:
    rewards_file.close()
    averages_file.close()
    vec_env.close()
