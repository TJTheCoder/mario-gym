import retro
import numpy as np
import torch
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Global variable to track if environment is already created
env_created = False

# Custom environment factory function
def make_env():
    global env_created
    if env_created:
        raise ValueError("Environment already created. Close it first.")
    env_created = True
    return retro.make(game="SuperMarioBros-Nes", state="Level1-1")

# Parameters
total_timesteps = 10_000_000  # Change this as needed
save_every = 10_000  # Save data every N frames

# Output files
os.makedirs("ppo_output", exist_ok=True)
rewards_file = open("ppo_output/rewards.txt", "w")
averages_file = open("ppo_output/averages.txt", "w")

# Buffers
rewards_buffer = []
average_buffer = []

# Create a single environment
env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")

# Wrap in DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Create PPO model
model = PPO("CnnPolicy", vec_env, verbose=1, device="auto")

# Train and collect data
n_steps = 0
start_time = time.time()

try:
    while n_steps < total_timesteps:
        model.learn(total_timesteps=save_every, reset_num_timesteps=False)
        
        # Evaluate policy
        obs = vec_env.reset()
        done = False
        total_reward = 0
        
        # Run one episode for evaluation
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
        
        # Save total return
        rewards_buffer.append(total_reward)
        
        # Calculate moving average (last 10 episodes or all if less than 10)
        moving_avg = np.mean(rewards_buffer[-10:])
        average_buffer.append(moving_avg)

        # Write to files
        rewards_file.write(f"{total_reward}\n")
        rewards_file.flush()
        averages_file.write(f"{moving_avg}\n")
        averages_file.flush()

        # Update step count
        n_steps += save_every
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print progress
        print(f"Step: {n_steps}/{total_timesteps} | Reward: {total_reward:.2f} | Moving Avg: {moving_avg:.2f} | Time: {elapsed_time:.1f}s")

        # Occasionally save the model
        if n_steps % 100000 == 0:
            model.save(f"ppo_output/mario_model_{n_steps}")

    # Final save
    model.save("ppo_output/mario_model_final")
    
    print("Training complete!")
    print(f"Final average reward (last 10 episodes): {moving_avg:.2f}")

finally:
    # Ensure cleanup happens even if there's an error
    rewards_file.close()
    averages_file.close()
    vec_env.close()
