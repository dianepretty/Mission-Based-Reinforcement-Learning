import gymnasium as gym
from stable_baselines3 import PPO, A2C
from environment.custom_env import CivicReportingEnv
import pandas as pd
import os

# Ensure the models directory exists
os.makedirs("models/pg", exist_ok=True)

# 10 experiments (5 PPO, 5 A2C) varying LR and Entropy Coefficient
# Entropy Coefficient (ent_coef) helps the agent explore different institutions
pg_experiments = [
    # PPO Experiments (Proximal Policy Optimization)
    {"algo": "PPO", "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99},
    {"algo": "PPO", "lr": 0.0003, "ent_coef": 0.01, "gamma": 0.99},
    {"algo": "PPO", "lr": 0.0001, "ent_coef": 0.05, "gamma": 0.95},
    {"algo": "PPO", "lr": 0.001, "ent_coef": 0.0, "gamma": 0.90},
    {"algo": "PPO", "lr": 0.0005, "ent_coef": 0.02, "gamma": 0.99},
    
    # A2C Experiments (Advantage Actor-Critic)
    {"algo": "A2C", "lr": 0.007, "ent_coef": 0.01, "gamma": 0.99},
    {"algo": "A2C", "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99},
    {"algo": "A2C", "lr": 0.0007, "ent_coef": 0.05, "gamma": 0.95},
    {"algo": "A2C", "lr": 0.0001, "ent_coef": 0.0, "gamma": 0.90},
    {"algo": "A2C", "lr": 0.0005, "ent_coef": 0.02, "gamma": 0.99},
]

pg_results = []

print("--- Starting Policy Gradient Training (PPO & A2C) ---")

for i, params in enumerate(pg_experiments):
    env = CivicReportingEnv()
    
    # Select Algorithm
    if params["algo"] == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, 
                    learning_rate=params['lr'], 
                    ent_coef=params['ent_coef'],
                    gamma=params['gamma'])
    else:
        model = A2C("MlpPolicy", env, verbose=0, 
                    learning_rate=params['lr'], 
                    ent_coef=params['ent_coef'],
                    gamma=params['gamma'])
    
    print(f"Exp {i+1}: {params['algo']} | LR={params['lr']} | Ent={params['ent_coef']} | Gamma={params['gamma']}")
    
    # Train for 50,000 steps for better visual curves in the report
    model.learn(total_timesteps=50000)
    
    # Save the model
    model_name = f"{params['algo'].lower()}_exp_{i+1}"
    model.save(f"models/pg/{model_name}")
    
    pg_results.append({**params, "Experiment": i+1, "Status": "Complete"})

# Save results to CSV for your Report Tables
df = pd.DataFrame(pg_results)
df.to_csv("pg_results.csv", index=False)

print("\n--- PG Training Done! Results saved to pg_results.csv ---")