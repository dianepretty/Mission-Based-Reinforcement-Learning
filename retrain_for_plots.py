"""
retrain_for_plots.py — Quick Retraining for Plot Generation
=============================================================
Retrains all 4 algorithms on the simplified environment
with enough episodes to show clear, positive learning curves.
Takes approximately 30-45 minutes total.

Delete this file after running.
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import CivicIssueEnv

TIMESTEPS = 200_000   # enough for clear learning curves
EPISODES  = 500       # for REINFORCE


# ── Reward tracker ────────────────────────────────────────────────────────────

class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = 0

    def _on_step(self):
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0
        return True


# ── SB3 training ──────────────────────────────────────────────────────────────

def train_sb3(algo_class, algo_name, configs, model_dir, log_dir):
    print(f"\n{'='*60}")
    print(f"  Training {algo_name} — {len(configs)} experiments")
    print(f"{'='*60}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    results = []

    for cfg in configs:
        print(f"\n  {cfg['name']} | LR={cfg['lr']} | γ={cfg['gamma']}")

        env     = Monitor(CivicIssueEnv())
        tracker = RewardTracker()

        if algo_class == DQN:
            model = DQN(
                "MlpPolicy", env,
                learning_rate        = cfg["lr"],
                gamma                = cfg["gamma"],
                buffer_size          = cfg.get("buffer_size", 50_000),
                batch_size           = cfg.get("batch_size", 64),
                exploration_fraction = cfg.get("exploration_fraction", 0.15),
                exploration_final_eps= 0.05,
                verbose=0
            )
        elif algo_class == PPO:
            model = PPO(
                "MlpPolicy", env,
                learning_rate = cfg["lr"],
                gamma         = cfg["gamma"],
                ent_coef      = cfg.get("ent_coef", 0.01),
                n_steps       = cfg.get("n_steps", 256),
                verbose=0
            )
        else:  # A2C
            model = A2C(
                "MlpPolicy", env,
                learning_rate = cfg["lr"],
                gamma         = cfg["gamma"],
                ent_coef      = cfg.get("ent_coef", 0.01),
                n_steps       = cfg.get("n_steps", 256),
                verbose=0
            )

        model.learn(total_timesteps=TIMESTEPS, callback=tracker, progress_bar=True)

        exp_dir = os.path.join(model_dir, cfg["name"])
        os.makedirs(exp_dir, exist_ok=True)
        model.save(os.path.join(exp_dir, "best_model"))

        rewards = tracker.episode_rewards
        np.save(os.path.join(log_dir, f"rewards_{cfg['name']}.npy"), np.array(rewards))

        # Save per-experiment npy in subfolder too
        exp_log = os.path.join(log_dir, cfg["name"])
        os.makedirs(exp_log, exist_ok=True)
        np.save(os.path.join(exp_log, "episode_rewards.npy"), np.array(rewards))

        final_mean = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)) if rewards else 0
        best       = float(np.max(rewards)) if rewards else 0

        conv = len(rewards)
        for i in range(20, len(rewards)):
            if np.mean(rewards[i-20:i]) > 0:
                conv = i
                break

        print(f"  Done | Final mean: {final_mean:.1f} | Best: {best:.1f} | Conv: {conv}")
        results.append({
            "Experiment": cfg["name"], "Learning Rate": cfg["lr"],
            "Gamma": cfg["gamma"], "Final Mean Reward": round(final_mean, 2),
            "Best Episode Reward": round(best, 2), "Convergence Episode": conv
        })
        env.close()

    # Save CSV
    csv_path = os.path.join(log_dir, f"{algo_name.lower()}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  ✅ {algo_name} complete — {csv_path}")
    return results


# ── REINFORCE ─────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)


def train_reinforce(configs, model_dir, log_dir):
    print(f"\n{'='*60}")
    print(f"  Training REINFORCE — {len(configs)} experiments")
    print(f"{'='*60}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    results = []

    for cfg in configs:
        print(f"\n  {cfg['name']} | LR={cfg['lr']} | γ={cfg['gamma']}")

        env     = CivicIssueEnv()
        policy  = PolicyNet(env.observation_space.shape[0], env.action_space.n, cfg.get("hidden", 128))
        opt     = optim.Adam(policy.parameters(), lr=cfg["lr"])
        rewards = []

        for ep in range(1, EPISODES + 1):
            obs, _ = env.reset()
            log_probs, ep_rewards = [], []
            done = False
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0)
                probs  = policy(obs_t)
                dist   = Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                obs, r, term, trunc, _ = env.step(action.item())
                ep_rewards.append(r)
                done = term or trunc

            # Compute returns
            G, returns = 0, []
            for r in reversed(ep_rewards):
                G = r + cfg["gamma"] * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            rewards.append(sum(ep_rewards))

            if ep % 100 == 0:
                print(f"    Ep {ep}/{EPISODES} | Rolling-50: {np.mean(rewards[-50:]):.1f} | Best: {max(rewards):.1f}")

        env.close()

        exp_dir = os.path.join(model_dir, cfg["name"])
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(exp_dir, "policy.pt"))

        exp_log = os.path.join(log_dir, cfg["name"])
        os.makedirs(exp_log, exist_ok=True)
        np.save(os.path.join(exp_log, "episode_rewards.npy"), np.array(rewards))

        final_mean = float(np.mean(rewards[-50:]))
        best       = float(np.max(rewards))
        conv = EPISODES
        for i in range(20, len(rewards)):
            if np.mean(rewards[i-20:i]) > 0:
                conv = i
                break

        print(f"  Done | Final mean: {final_mean:.1f} | Best: {best:.1f} | Conv: {conv}")
        results.append({
            "Experiment": cfg["name"], "Learning Rate": cfg["lr"],
            "Gamma": cfg["gamma"], "Final Mean Reward": round(final_mean, 2),
            "Best Episode Reward": round(best, 2), "Convergence Episode": conv
        })

    csv_path = os.path.join(log_dir, "reinforce_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  ✅ REINFORCE complete — {csv_path}")
    return results


# ── Configs ───────────────────────────────────────────────────────────────────

dqn_configs = [
    {"name":"Exp01_Baseline",       "lr":1e-3,  "gamma":0.99, "buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp02_LowLR",          "lr":1e-4,  "gamma":0.99, "buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp03_HighLR",         "lr":1e-2,  "gamma":0.99, "buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp04_ShortSighted",   "lr":1e-3,  "gamma":0.80, "buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp05_LongSighted",    "lr":1e-3,  "gamma":0.999,"buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp06_LargeBuffer",    "lr":1e-3,  "gamma":0.99, "buffer_size":200_000, "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp07_SmallBuffer",    "lr":1e-3,  "gamma":0.99, "buffer_size":10_000,  "batch_size":64,  "exploration_fraction":0.15},
    {"name":"Exp08_LargeBatch",     "lr":1e-3,  "gamma":0.99, "buffer_size":50_000,  "batch_size":256, "exploration_fraction":0.15},
    {"name":"Exp09_HighExploration","lr":1e-3,  "gamma":0.99, "buffer_size":50_000,  "batch_size":64,  "exploration_fraction":0.40},
    {"name":"Exp10_Balanced",       "lr":3e-4,  "gamma":0.95, "buffer_size":100_000, "batch_size":128, "exploration_fraction":0.20},
]

pg_configs = [
    {"name":"Exp01_Baseline",     "lr":3e-4, "gamma":0.99,  "ent_coef":0.01, "n_steps":256},
    {"name":"Exp02_HighLR",       "lr":1e-3, "gamma":0.99,  "ent_coef":0.01, "n_steps":256},
    {"name":"Exp03_LowLR",        "lr":1e-4, "gamma":0.99,  "ent_coef":0.01, "n_steps":256},
    {"name":"Exp04_ShortSighted", "lr":3e-4, "gamma":0.80,  "ent_coef":0.01, "n_steps":256},
    {"name":"Exp05_LongSighted",  "lr":3e-4, "gamma":0.999, "ent_coef":0.01, "n_steps":256},
    {"name":"Exp06_HighEntropy",  "lr":3e-4, "gamma":0.99,  "ent_coef":0.05, "n_steps":256},
    {"name":"Exp07_LowEntropy",   "lr":3e-4, "gamma":0.99,  "ent_coef":0.001,"n_steps":256},
    {"name":"Exp08_LargeSteps",   "lr":3e-4, "gamma":0.99,  "ent_coef":0.01, "n_steps":512},
    {"name":"Exp09_SmallSteps",   "lr":3e-4, "gamma":0.99,  "ent_coef":0.01, "n_steps":128},
    {"name":"Exp10_Balanced",     "lr":5e-4, "gamma":0.95,  "ent_coef":0.02, "n_steps":256},
]

rf_configs = [
    {"name":"Exp01_Baseline",     "lr":1e-3,   "gamma":0.99,  "hidden":128},
    {"name":"Exp02_HighLR",       "lr":5e-3,   "gamma":0.99,  "hidden":128},
    {"name":"Exp03_LowLR",        "lr":1e-4,   "gamma":0.99,  "hidden":128},
    {"name":"Exp04_ShortSighted", "lr":1e-3,   "gamma":0.80,  "hidden":128},
    {"name":"Exp05_LongSighted",  "lr":1e-3,   "gamma":0.999, "hidden":128},
    {"name":"Exp06_LargeNet",     "lr":1e-3,   "gamma":0.99,  "hidden":256},
    {"name":"Exp07_SmallNet",     "lr":1e-3,   "gamma":0.99,  "hidden":64},
    {"name":"Exp08_Balanced",     "lr":5e-4,   "gamma":0.95,  "hidden":128},
    {"name":"Exp09_Aggressive",   "lr":1e-2,   "gamma":0.99,  "hidden":128},
    {"name":"Exp10_Conservative", "lr":5e-5,   "gamma":0.99,  "hidden":128},
]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RETRAINING ALL ALGORITHMS — Simplified Environment")
    print("  Estimated time: 30-45 minutes")
    print("="*60)

    train_sb3(DQN, "DQN", dqn_configs, "models/dqn", "logs/dqn")
    train_sb3(PPO, "PPO", pg_configs,  "models/ppo", "logs/ppo")
    train_sb3(A2C, "A2C", pg_configs,  "models/a2c", "logs/a2c")
    train_reinforce(rf_configs, "models/reinforce", "logs/reinforce")

    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE!")
    print("  Now run: python generate_plots.py")
    print("  Then delete this file: del retrain_for_plots.py")
    print("="*60 + "\n")