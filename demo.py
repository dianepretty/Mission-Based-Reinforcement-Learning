# main.py
import time
from environment.custom_env import CivicReportingEnv
from environment.rendering import CivicRenderer

env = CivicReportingEnv()
renderer = CivicRenderer(env.institutions)

obs, _ = env.reset()
total_reward = 0

print("Starting Random Action Simulation for Visualization...")

for _ in range(30): # Run for 30 steps
    action = env.action_space.sample() # RANDOM ACTIONS
    obs, reward, term, trunc, _ = env.step(action)
    total_reward += reward
    
    renderer.render(obs, action, total_reward)
    time.sleep(0.5) # Slow down so we can see it

renderer.close()
print("Visualization Complete.")