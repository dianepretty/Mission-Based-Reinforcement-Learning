import gymnasium as gym
from stable_baselines3 import DQN
from environment.custom_env import CivicReportingEnv
import pandas as pd

# The 10 experiments with varying learning rates and discount factors
experiments = [
  {"lr": 1e-2, "gamma": 0.99}, 
    {"lr": 5e-3, "gamma": 0.99},
    {"lr": 1e-3, "gamma": 0.99}, # Usually the "sweet spot"
    {"lr": 5e-4, "gamma": 0.99},
    {"lr": 1e-4, "gamma": 0.99},

    # Group 2: Testing Gamma (Short-term vs. Long-term accountability)
    {"lr": 1e-3, "gamma": 0.70}, # Very shortsighted
    {"lr": 1e-3, "gamma": 0.85}, 
    {"lr": 1e-3, "gamma": 0.90},
    {"lr": 1e-3, "gamma": 0.95},
    {"lr": 1e-3, "gamma": 0.995} # Extremely farsighted
]

results = []

print("--- Starting DQN Training Experiments ---")

for i, params in enumerate(experiments):
    env = CivicReportingEnv()
    # Initialize the "Brain"
    model = DQN("MlpPolicy", env, verbose=0, 
                learning_rate=params['lr'], 
                gamma=params['gamma'])
    
    print(f"Running Exp {i+1}: LR={params['lr']}, Gamma={params['gamma']}")
    model.learn(total_timesteps=50000)
    
    # Save the model
    model.save(f"models/dqn/dqn_exp_{i+1}")
    
    # Simple check for the report: record the final performance
    results.append({
        "Experiment": i+1,
        "LR": params['lr'],
        "Gamma": params['gamma'],
        "Status": "Complete"
    })

# Save results for your report table
pd.DataFrame(results).to_csv("dqn_results.csv")
print("--- Training Done! Check dqn_results.csv for your data. ---")