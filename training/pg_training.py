"""
Policy Gradient Training Script — Rwanda Civic Issue RL
========================================================
Trains PPO and A2C (10 experiments each = 20 total) using Stable Baselines 3.
Both algorithms share the same environment and experiment configs for
a fair apples-to-apples comparison.

Hyperparameters varied:
  - learning_rate  : step size for policy updates
  - gamma          : discount factor
  - ent_coef       : entropy coefficient (exploration bonus)
  - n_steps        : steps collected before each update
  - clip_range     : PPO clipping parameter (PPO only)
  - gae_lambda     : GAE smoothing factor
"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import CivicIssueEnv

# ── Directories ───────────────────────────────────────────────────────────────
PPO_MODEL_DIR = os.path.join("models", "ppo")
A2C_MODEL_DIR = os.path.join("models", "a2c")
PPO_LOG_DIR   = os.path.join("logs",   "ppo")
A2C_LOG_DIR   = os.path.join("logs",   "a2c")

for d in [PPO_MODEL_DIR, A2C_MODEL_DIR, PPO_LOG_DIR, A2C_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 150_000


# ── Reward tracker ────────────────────────────────────────────────────────────

class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_lengths  = []
        self._ep_reward       = 0
        self._ep_length       = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_length += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self._ep_reward = 0
            self._ep_length = 0
        return True


# ── 10 Shared experiment configurations ──────────────────────────────────────
# Both PPO and A2C run these same configs for direct comparison.

experiments = [
    {
        "name":        "Exp01_Baseline",
        "lr":          3e-4,
        "gamma":       0.99,
        "ent_coef":    0.01,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Standard baseline — default SB3 settings"
    },
    {
        "name":        "Exp02_HighLR",
        "lr":          1e-3,
        "gamma":       0.99,
        "ent_coef":    0.01,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "High learning rate — faster but noisier updates"
    },
    {
        "name":        "Exp03_LowLR",
        "lr":          1e-4,
        "gamma":       0.99,
        "ent_coef":    0.01,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Low learning rate — slow stable convergence"
    },
    {
        "name":        "Exp04_ShortSighted",
        "lr":          3e-4,
        "gamma":       0.80,
        "ent_coef":    0.01,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Low gamma — agent focuses on immediate dispatch reward"
    },
    {
        "name":        "Exp05_LongSighted",
        "lr":          3e-4,
        "gamma":       0.999,
        "ent_coef":    0.01,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Very high gamma — long term trust building prioritised"
    },
    {
        "name":        "Exp06_HighEntropy",
        "lr":          3e-4,
        "gamma":       0.99,
        "ent_coef":    0.05,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "High entropy — agent explores more action variety"
    },
    {
        "name":        "Exp07_LowEntropy",
        "lr":          3e-4,
        "gamma":       0.99,
        "ent_coef":    0.001,
        "n_steps":     512,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Low entropy — agent commits to learned policy early"
    },
    {
        "name":        "Exp08_LargeSteps",
        "lr":          3e-4,
        "gamma":       0.99,
        "ent_coef":    0.01,
        "n_steps":     1024,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Large n_steps — more data per update, better gradient"
    },
    {
        "name":        "Exp09_SmallSteps",
        "lr":          3e-4,
        "gamma":       0.99,
        "ent_coef":    0.01,
        "n_steps":     128,
        "gae_lambda":  0.95,
        "clip_range":  0.2,
        "description": "Small n_steps — frequent updates, more reactive"
    },
    {
        "name":        "Exp10_Balanced",
        "lr":          5e-4,
        "gamma":       0.95,
        "ent_coef":    0.02,
        "n_steps":     256,
        "gae_lambda":  0.90,
        "clip_range":  0.15,
        "description": "Balanced — tuned for civic env dynamics"
    },
]


# ── Training function ─────────────────────────────────────────────────────────

def train_experiment(algo_class, algo_name: str, cfg: dict,
                     model_dir: str, log_dir: str) -> dict:

    print(f"\n{'='*65}")
    print(f"  {algo_name} | {cfg['name']}")
    print(f"  LR={cfg['lr']} | γ={cfg['gamma']} | "
          f"ent={cfg['ent_coef']} | n_steps={cfg['n_steps']}")
    print(f"  {cfg['description']}")
    print(f"{'='*65}")

    env      = Monitor(CivicIssueEnv())
    eval_env = Monitor(CivicIssueEnv())
    tracker  = RewardTrackerCallback()

    run_name = f"{algo_name}_{cfg['name']}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(model_dir, cfg["name"]),
        log_path             = os.path.join(log_dir,   cfg["name"]),
        eval_freq            = 10_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
    )

    # Build model — PPO has clip_range, A2C does not
    if algo_class == PPO:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = cfg["lr"],
            gamma         = cfg["gamma"],
            ent_coef      = cfg["ent_coef"],
            n_steps       = cfg["n_steps"],
            gae_lambda    = cfg["gae_lambda"],
            clip_range    = cfg["clip_range"],
            tensorboard_log = log_dir,
            verbose       = 0,
        )
    else:
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate = cfg["lr"],
            gamma         = cfg["gamma"],
            ent_coef      = cfg["ent_coef"],
            n_steps       = cfg["n_steps"],
            gae_lambda    = cfg["gae_lambda"],
            tensorboard_log = log_dir,
            verbose       = 0,
        )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [tracker, eval_callback],
        tb_log_name     = run_name,
        progress_bar    = True,
    )

    model.save(os.path.join(model_dir, cfg["name"], "final_model"))

    # Save rewards
    rewards_path = os.path.join(log_dir, cfg["name"], "episode_rewards.npy")
    os.makedirs(os.path.dirname(rewards_path), exist_ok=True)
    np.save(rewards_path, np.array(tracker.episode_rewards))

    env.close()
    eval_env.close()

    rewards    = tracker.episode_rewards
    final_mean = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards)) if rewards else 0.0
    best_ep    = float(np.max(rewards)) if rewards else 0.0

    convergence_ep = len(rewards)
    for i in range(20, len(rewards)):
        if np.mean(rewards[i-20:i]) > 0:
            convergence_ep = i
            break

    print(f"\n  ✓ Done | Final mean: {final_mean:.2f} | Best: {best_ep:.2f} | Conv.ep: {convergence_ep}")

    return {
        "Algorithm":           algo_name,
        "Experiment":          cfg["name"],
        "Learning Rate":       cfg["lr"],
        "Gamma":               cfg["gamma"],
        "Entropy Coef":        cfg["ent_coef"],
        "N Steps":             cfg["n_steps"],
        "GAE Lambda":          cfg["gae_lambda"],
        "Clip Range":          cfg["clip_range"] if algo_class == PPO else "N/A",
        "Total Episodes":      len(rewards),
        "Final Mean Reward":   round(final_mean, 2),
        "Best Episode Reward": round(best_ep, 2),
        "Convergence Episode": convergence_ep,
        "Description":         cfg["description"],
    }


# ── Run all experiments ───────────────────────────────────────────────────────

def run_ppo():
    print("\n" + "="*65)
    print("  PPO EXPERIMENTS — Rwanda Civic Dispatch RL")
    print("="*65)

    results = []
    for cfg in experiments:
        r = train_experiment(PPO, "PPO", cfg, PPO_MODEL_DIR, PPO_LOG_DIR)
        results.append(r)

    csv_path = os.path.join(PPO_LOG_DIR, "ppo_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ PPO complete — results: {csv_path}")
    return results


def run_a2c():
    print("\n" + "="*65)
    print("  A2C EXPERIMENTS — Rwanda Civic Dispatch RL")
    print("="*65)

    results = []
    for cfg in experiments:
        r = train_experiment(A2C, "A2C", cfg, A2C_MODEL_DIR, A2C_LOG_DIR)
        results.append(r)

    csv_path = os.path.join(A2C_LOG_DIR, "a2c_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ A2C complete — results: {csv_path}")
    return results


def run_all():
    ppo_results = run_ppo()
    a2c_results = run_a2c()

    print("\n" + "="*65)
    print("  ✅ ALL POLICY GRADIENT EXPERIMENTS COMPLETE")
    print(f"  PPO models : {PPO_MODEL_DIR}/")
    print(f"  A2C models : {A2C_MODEL_DIR}/")
    print("="*65)

    return ppo_results, a2c_results


if __name__ == "__main__":
    run_all()