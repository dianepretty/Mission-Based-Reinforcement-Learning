"""
DQN Training Script — Rwanda Civic Issue RL
============================================
Trains 10 DQN experiments with varied hyperparameters using Stable Baselines 3.
Results are saved to logs/dqn/ and models/dqn/ for analysis and report tables.

Hyperparameters varied:
  - learning_rate    : how fast the network updates
  - gamma            : discount factor (short vs long term thinking)
  - buffer_size      : experience replay memory size
  - batch_size       : samples per gradient update
  - exploration      : epsilon-greedy exploration fraction
"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import CivicIssueEnv

# ── Directories ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join("models", "dqn")
LOG_DIR   = os.path.join("logs",   "dqn")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 150_000   # enough for convergence on this env


# ── Reward tracker callback ───────────────────────────────────────────────────

class RewardTrackerCallback(BaseCallback):
    """Tracks episode rewards during training for post-analysis."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_lengths  = []
        self._current_rewards = 0
        self._current_length  = 0

    def _on_step(self) -> bool:
        self._current_rewards += self.locals["rewards"][0]
        self._current_length  += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_rewards)
            self.episode_lengths.append(self._current_length)
            self._current_rewards = 0
            self._current_length  = 0
        return True


# ── 10 Experiment configurations ──────────────────────────────────────────────
# Each row varies meaningful hyperparameters to show different learning behavior.
# This gives the report table real data to discuss.

experiments = [
    {
        "name":            "Exp01_Baseline",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Standard baseline — default SB3 settings"
    },
    {
        "name":            "Exp02_LowLR",
        "learning_rate":   1e-4,
        "gamma":           0.99,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Lower LR — slower but more stable convergence"
    },
    {
        "name":            "Exp03_HighLR",
        "learning_rate":   1e-2,
        "gamma":           0.99,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "High LR — fast but risks instability"
    },
    {
        "name":            "Exp04_ShortSighted",
        "learning_rate":   1e-3,
        "gamma":           0.80,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Low gamma — agent prioritises immediate rewards"
    },
    {
        "name":            "Exp05_LongSighted",
        "learning_rate":   1e-3,
        "gamma":           0.999,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Very high gamma — agent plans far into the future"
    },
    {
        "name":            "Exp06_LargeBuffer",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "buffer_size":     200_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Large replay buffer — more diverse experience"
    },
    {
        "name":            "Exp07_SmallBuffer",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "buffer_size":     10_000,
        "batch_size":      64,
        "exploration_fraction": 0.15,
        "description":     "Small buffer — agent forgets old experience quickly"
    },
    {
        "name":            "Exp08_LargeBatch",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "buffer_size":     50_000,
        "batch_size":      256,
        "exploration_fraction": 0.15,
        "description":     "Large batch — more stable gradient estimates"
    },
    {
        "name":            "Exp09_HighExploration",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "buffer_size":     50_000,
        "batch_size":      64,
        "exploration_fraction": 0.40,
        "description":     "High exploration — agent tries more random actions longer"
    },
    {
        "name":            "Exp10_Balanced",
        "learning_rate":   3e-4,
        "gamma":           0.95,
        "buffer_size":     100_000,
        "batch_size":      128,
        "exploration_fraction": 0.20,
        "description":     "Balanced tuning — best of all worlds"
    },
]


# ── Training function ─────────────────────────────────────────────────────────

def train_experiment(cfg: dict) -> dict:
    """Train one DQN experiment and return result metrics."""

    print(f"\n{'='*65}")
    print(f"  DQN | {cfg['name']}")
    print(f"  LR={cfg['learning_rate']} | γ={cfg['gamma']} | "
          f"buf={cfg['buffer_size']:,} | batch={cfg['batch_size']} | "
          f"explore={cfg['exploration_fraction']}")
    print(f"  {cfg['description']}")
    print(f"{'='*65}")

    # Wrap env in Monitor for SB3 episode tracking
    env      = Monitor(CivicIssueEnv())
    eval_env = Monitor(CivicIssueEnv())

    # Reward tracker
    tracker = RewardTrackerCallback()

    # Best model saver
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, cfg["name"]),
        log_path=os.path.join(LOG_DIR, cfg["name"]),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate       = cfg["learning_rate"],
        gamma               = cfg["gamma"],
        buffer_size         = cfg["buffer_size"],
        batch_size          = cfg["batch_size"],
        exploration_fraction= cfg["exploration_fraction"],
        exploration_final_eps=0.05,
        train_freq          = 4,
        target_update_interval=1000,
        tensorboard_log     = LOG_DIR,
        verbose             = 0,
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [tracker, eval_callback],
        tb_log_name     = cfg["name"],
        progress_bar    = True,
    )

    # Save final model
    model.save(os.path.join(MODEL_DIR, cfg["name"], "final_model"))

    # Save reward history for plotting
    rewards_path = os.path.join(LOG_DIR, cfg["name"], "episode_rewards.npy")
    os.makedirs(os.path.dirname(rewards_path), exist_ok=True)
    np.save(rewards_path, np.array(tracker.episode_rewards))

    env.close()
    eval_env.close()

    # Compute metrics
    rewards = tracker.episode_rewards
    final_mean = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards)) if rewards else 0.0
    best_reward = float(np.max(rewards)) if rewards else 0.0

    # Convergence: first episode where rolling-20 mean > 0
    convergence_ep = len(rewards)
    for i in range(20, len(rewards)):
        if np.mean(rewards[i-20:i]) > 0:
            convergence_ep = i
            break

    print(f"\n  ✓ Done | Final mean reward (last 20 eps): {final_mean:.2f}")
    print(f"         Best episode reward: {best_reward:.2f}")
    print(f"         Convergence episode: {convergence_ep}")

    return {
        "Experiment":          cfg["name"],
        "Learning Rate":       cfg["learning_rate"],
        "Gamma":               cfg["gamma"],
        "Buffer Size":         cfg["buffer_size"],
        "Batch Size":          cfg["batch_size"],
        "Exploration Fraction":cfg["exploration_fraction"],
        "Total Episodes":      len(rewards),
        "Final Mean Reward":   round(final_mean, 2),
        "Best Episode Reward": round(best_reward, 2),
        "Convergence Episode": convergence_ep,
        "Description":         cfg["description"],
    }


# ── Run all experiments ───────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*65)
    print("  DQN HYPERPARAMETER EXPERIMENTS — Rwanda Civic Dispatch RL")
    print("="*65)

    all_results = []

    for cfg in experiments:
        result = train_experiment(cfg)
        all_results.append(result)

    # Save results CSV for report table
    csv_path = os.path.join(LOG_DIR, "dqn_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*65}")
    print(f"  ✅ All 10 DQN experiments complete!")
    print(f"  Results CSV  : {csv_path}")
    print(f"  Models saved : {MODEL_DIR}/")
    print(f"  Logs saved   : {LOG_DIR}/")
    print(f"{'='*65}")

    # Print summary table
    print(f"\n{'Experiment':<25} {'LR':<8} {'Gamma':<7} {'Final Reward':<14} {'Best':<10} {'Conv.Ep'}")
    print("-" * 80)
    for r in all_results:
        print(f"  {r['Experiment']:<23} {r['Learning Rate']:<8} "
              f"{r['Gamma']:<7} {r['Final Mean Reward']:<14} "
              f"{r['Best Episode Reward']:<10} {r['Convergence Episode']}")

    return all_results


if __name__ == "__main__":
    run_all()   