"""
train_best.py — Retrain Best DQN Model
========================================
Retrains the best DQN configuration at 500,000 timesteps.
Delete this file after training is complete.

Usage:
    python train_best.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from environment.custom_env import CivicIssueEnv


def train():
    print("\n" + "="*60)
    print("  Retraining DQN — Exp06_LargeBuffer")
    print("  LR=1e-3 | Gamma=0.99 | Buffer=200k | Batch=64")
    print("  Estimated time: 20-30 minutes")
    print("="*60)

    os.makedirs("models/dqn/Exp06_LargeBuffer", exist_ok=True)
    os.makedirs("logs/dqn/Exp06_LargeBuffer",   exist_ok=True)

    env      = Monitor(CivicIssueEnv())
    eval_env = Monitor(CivicIssueEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = "models/dqn/Exp06_LargeBuffer",
        log_path             = "logs/dqn/Exp06_LargeBuffer",
        eval_freq            = 10_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate         = 1e-3,
        gamma                 = 0.99,
        buffer_size           = 200_000,
        batch_size            = 64,
        exploration_fraction  = 0.15,
        exploration_final_eps = 0.05,
        train_freq            = 4,
        target_update_interval= 1000,
        verbose               = 0,
    )

    model.learn(
        total_timesteps = 500_000,
        callback        = eval_callback,
        progress_bar    = True,
    )

    model.save("models/dqn/Exp06_LargeBuffer/final_model")
    env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("  ✅ Model saved to models/dqn/Exp06_LargeBuffer/")
    print("\n  Now run:")
    print("    python main.py --algo dqn")
    print("\n  Then delete this file:")
    print("    del train_best.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()