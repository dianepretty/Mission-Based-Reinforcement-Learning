import gymnasium as gym
from environment.custom_env import CivicReportingEnv
import time

# 1. Create the environment
env = CivicReportingEnv(render_mode="human")
obs, info = env.reset()

print("--- Starting Random Action Test ---")

# 2. Run for 20 steps with random actions
for step in range(20):
    # Select a random action (0, 1, 2, or 3)
    action = env.action_space.sample()
    
    # Apply the action to the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print the result to the terminal
    env.render()
    
    # Pause slightly so we can read the output
    time.sleep(0.5)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("--- Test Complete ---")