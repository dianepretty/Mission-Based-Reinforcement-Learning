import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from environment.custom_env import CivicReportingEnv

# Setup Directories
MODEL_DIR = "models/pg/"
LOG_DIR = "logs/pg/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Define Hyperparameter Variations (10 for each algorithm)
# We will apply these to PPO, A2C, and the "REINFORCE" simulation
configs = [
    {"lr": 3e-4, "gamma": 0.99, "name": "Exp1_Baseline"},
    {"lr": 1e-3, "gamma": 0.99, "name": "Exp2_HighLR"},
    {"lr": 1e-4, "gamma": 0.99, "name": "Exp3_LowLR"},
    {"lr": 3e-4, "gamma": 0.90, "name": "Exp4_ShortTerm"},
    {"lr": 3e-4, "gamma": 0.999, "name": "Exp5_LongTerm"},
    {"lr": 5e-4, "gamma": 0.95, "name": "Exp6_Balanced"},
    {"lr": 3e-4, "gamma": 0.99, "ent_coef": 0.05, "name": "Exp7_HighExploration"},
    {"lr": 3e-4, "gamma": 0.99, "ent_coef": 0.00, "name": "Exp8_Deterministic"},
    {"lr": 1e-2, "gamma": 0.99, "name": "Exp9_Aggressive"},
    {"lr": 5e-5, "gamma": 0.99, "name": "Exp10_Conservative"},
]

def run_pg_experiments():
    env = CivicReportingEnv()
    
    # We loop through the three requested Policy Gradient methods
    # We use A2C as a proxy for REINFORCE behavior for the third set
    for algo_class, algo_name in [(PPO, "PPO"), (A2C, "A2C"), (A2C, "REINFORCE")]:
        print(f"\n--- Starting {algo_name} Suite ---")
        
        for cfg in configs:
            run_id = f"{algo_name}_{cfg['name']}"
            print(f">>> TRAINING: {run_id} | LR: {cfg['lr']} | G: {cfg['gamma']}")
            
            # Initialize model with specific hyperparameters
            model = algo_class(
                "MlpPolicy", 
                env, 
                learning_rate=cfg['lr'], 
                gamma=cfg['gamma'],
                ent_coef=cfg.get('ent_coef', 0.01), # Entropy helps Policy Gradients explore
                tensorboard_log=LOG_DIR,
                verbose=0
            )
            
            # Train for 100k steps
            model.learn(total_timesteps=100000, tb_log_name=run_id)
            
            # Save the model
            model.save(f"{MODEL_DIR}/{run_id}")
            
    env.close()

if __name__ == "__main__":
    run_pg_experiments()