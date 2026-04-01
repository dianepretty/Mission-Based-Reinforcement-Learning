import os
import gymnasium as gym
from stable_baselines3 import DQN
from environment.custom_env import CivicReportingEnv

# Setup Directories
MODEL_DIR = "models/dqn/"
LOG_DIR = "logs/dqn/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Define your 10 Experiment Configurations
# We vary Learning Rate (LR) and Gamma (G) to see how the agent prioritizes
experiments = [
    {"lr": 1e-3, "gamma": 0.99, "name": "Exp1_Baseline"},
    {"lr": 5e-4, "gamma": 0.99, "name": "Exp2_LowerLR"},
    {"lr": 1e-4, "gamma": 0.99, "name": "Exp3_VeryLowLR"},
    {"lr": 1e-3, "gamma": 0.95, "name": "Exp4_ShortTerm"},
    {"lr": 1e-3, "gamma": 0.80, "name": "Exp5_ImmediateOnly"},
    {"lr": 5e-4, "gamma": 0.95, "name": "Exp6_Balanced"},
    {"lr": 1e-3, "gamma": 0.999, "name": "Exp7_LongTerm"},
    {"lr": 3e-4, "gamma": 0.99, "name": "Exp8_StableBaselinesDefault"},
    {"lr": 1e-2, "gamma": 0.99, "name": "Exp9_HighLR_Aggressive"},
    {"lr": 1e-3, "gamma": 0.50, "name": "Exp10_Myopic_Agent"},
]

def run_experiments():
    env = CivicReportingEnv()
    
    for exp in experiments:
        print(f"\n>>> RUNNING: {exp['name']} | LR: {exp['lr']} | Gamma: {exp['gamma']}")
        
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0, # Set to 0 to keep terminal clean during 10 runs
            learning_rate=exp['lr'],
            gamma=exp['gamma'],
            tensorboard_log=LOG_DIR
        )
        
        # Train for 100k steps
        model.learn(total_timesteps=100000, tb_log_name=exp['name'])
        
        # Save each model uniquely
        model.save(f"{MODEL_DIR}/dqn_{exp['name']}")
        
        print(f">>> FINISHED: {exp['name']}")

    env.close()

if __name__ == "__main__":
    run_experiments()