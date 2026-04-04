"""
REINFORCE Training Script — Rwanda Civic Issue RL
==================================================
Pure Monte Carlo Policy Gradient (REINFORCE) implemented from scratch
in PyTorch. Unlike DQN/PPO/A2C which use Stable Baselines 3, this is
written manually to demonstrate understanding of the core algorithm.

How REINFORCE works:
  1. Run a full episode collecting (state, action, reward) at every step
  2. Compute discounted return G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
  3. Update policy: loss = -Σ log π(a_t|s_t) × G_t
  4. Normalise returns for variance reduction
  5. Repeat for N episodes

Key difference from A2C/PPO:
  - No value function baseline (pure policy gradient)
  - Full episode before any update (Monte Carlo, not TD)
  - Higher variance but conceptually clean
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.custom_env import CivicIssueEnv

# ── Directories ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join("models", "reinforce")
LOG_DIR   = os.path.join("logs",   "reinforce")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

EPISODES = 1000   # REINFORCE needs more episodes (no replay buffer)


# ── Policy Network ────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    2-layer MLP policy network.
    Input  : 60-dim observation vector
    Output : softmax over 12 discrete actions
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Core REINFORCE functions ──────────────────────────────────────────────────

def select_action(policy: PolicyNetwork, obs: np.ndarray):
    """Sample action from policy, return (action, log_prob)."""
    obs_t  = torch.FloatTensor(obs).unsqueeze(0)
    probs  = policy(obs_t)
    dist   = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns G_t for each timestep.
    Normalise for training stability (variance reduction).
    """
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def run_episode(env: CivicIssueEnv, policy: PolicyNetwork):
    """Collect one complete episode. Returns (log_probs, rewards)."""
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        action, log_prob = select_action(policy, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        done = terminated or truncated

    return log_probs, rewards


def update_policy(optimizer: optim.Optimizer,
                  log_probs: list,
                  returns: torch.Tensor) -> float:
    """
    REINFORCE gradient update.
    Loss = -Σ log π(a_t|s_t) × G_t
    Minimising this maximises expected return.
    """
    loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── 10 Experiment configurations ──────────────────────────────────────────────

experiments = [
    {
        "name":        "Exp01_Baseline",
        "lr":          1e-3,
        "gamma":       0.99,
        "hidden":      128,
        "description": "Standard baseline"
    },
    {
        "name":        "Exp02_HighLR",
        "lr":          5e-3,
        "gamma":       0.99,
        "hidden":      128,
        "description": "High LR — fast but unstable"
    },
    {
        "name":        "Exp03_LowLR",
        "lr":          1e-4,
        "gamma":       0.99,
        "hidden":      128,
        "description": "Low LR — slow stable learning"
    },
    {
        "name":        "Exp04_ShortSighted",
        "lr":          1e-3,
        "gamma":       0.80,
        "hidden":      128,
        "description": "Low gamma — immediate reward focus"
    },
    {
        "name":        "Exp05_LongSighted",
        "lr":          1e-3,
        "gamma":       0.999,
        "hidden":      128,
        "description": "Very high gamma — long term planning"
    },
    {
        "name":        "Exp06_LargeNet",
        "lr":          1e-3,
        "gamma":       0.99,
        "hidden":      256,
        "description": "Larger network — more capacity"
    },
    {
        "name":        "Exp07_SmallNet",
        "lr":          1e-3,
        "gamma":       0.99,
        "hidden":      64,
        "description": "Smaller network — less capacity, faster"
    },
    {
        "name":        "Exp08_Balanced",
        "lr":          5e-4,
        "gamma":       0.95,
        "hidden":      128,
        "description": "Balanced LR and gamma"
    },
    {
        "name":        "Exp09_Aggressive",
        "lr":          1e-2,
        "gamma":       0.99,
        "hidden":      128,
        "description": "Very high LR — tests instability limits"
    },
    {
        "name":        "Exp10_Conservative",
        "lr":          5e-5,
        "gamma":       0.99,
        "hidden":      128,
        "description": "Very low LR — extremely cautious updates"
    },
]


# ── Training function ─────────────────────────────────────────────────────────

def train_experiment(cfg: dict) -> dict:
    print(f"\n{'='*65}")
    print(f"  REINFORCE | {cfg['name']}")
    print(f"  LR={cfg['lr']} | γ={cfg['gamma']} | hidden={cfg['hidden']}")
    print(f"  {cfg['description']}")
    print(f"{'='*65}")

    env     = CivicIssueEnv()
    obs_dim = env.observation_space.shape[0]   # 60
    act_dim = env.action_space.n               # 12

    policy    = PolicyNetwork(obs_dim, act_dim, cfg["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])

    episode_rewards = []
    best_reward     = -np.inf
    convergence_ep  = EPISODES

    for ep in range(1, EPISODES + 1):
        log_probs, rewards = run_episode(env, policy)
        returns            = compute_returns(rewards, cfg["gamma"])
        update_policy(optimizer, log_probs, returns)

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward

        # Convergence check: rolling-50 mean > 0
        if ep >= 50 and convergence_ep == EPISODES:
            if np.mean(episode_rewards[-50:]) > 0:
                convergence_ep = ep

        if ep % 100 == 0:
            rolling = np.mean(episode_rewards[-50:])
            print(f"  Ep {ep:04d}/{EPISODES} | "
                  f"Rolling-50: {rolling:+.2f} | "
                  f"Best: {best_reward:+.2f}")

    # Save model and rewards
    exp_model_dir = os.path.join(MODEL_DIR, cfg["name"])
    os.makedirs(exp_model_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(exp_model_dir, "policy.pt"))

    exp_log_dir = os.path.join(LOG_DIR, cfg["name"])
    os.makedirs(exp_log_dir, exist_ok=True)
    np.save(os.path.join(exp_log_dir, "episode_rewards.npy"),
            np.array(episode_rewards))

    env.close()

    final_mean = float(np.mean(episode_rewards[-50:]))

    print(f"\n  ✓ Done | Final mean (last 50 eps): {final_mean:.2f} | "
          f"Best: {best_reward:.2f} | Conv.ep: {convergence_ep}")

    return {
        "Experiment":          cfg["name"],
        "Learning Rate":       cfg["lr"],
        "Gamma":               cfg["gamma"],
        "Hidden Size":         cfg["hidden"],
        "Episodes":            EPISODES,
        "Final Mean Reward":   round(final_mean, 2),
        "Best Episode Reward": round(best_reward, 2),
        "Convergence Episode": convergence_ep,
        "Description":         cfg["description"],
    }


# ── Run all experiments ───────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*65)
    print("  REINFORCE EXPERIMENTS — Rwanda Civic Dispatch RL")
    print("="*65)

    all_results = []
    for cfg in experiments:
        result = train_experiment(cfg)
        all_results.append(result)

    csv_path = os.path.join(LOG_DIR, "reinforce_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*65}")
    print(f"  ✅ All 10 REINFORCE experiments complete!")
    print(f"  Results CSV  : {csv_path}")
    print(f"  Models saved : {MODEL_DIR}/")
    print(f"{'='*65}\n")

    print(f"{'Experiment':<25} {'LR':<8} {'Gamma':<7} {'Final Mean':<14} {'Best':<10} {'Conv.Ep'}")
    print("-" * 78)
    for r in all_results:
        print(f"  {r['Experiment']:<23} {r['Learning Rate']:<8} "
              f"{r['Gamma']:<7} {r['Final Mean Reward']:<14} "
              f"{r['Best Episode Reward']:<10} {r['Convergence Episode']}")

    return all_results


if __name__ == "__main__":
    run_all()