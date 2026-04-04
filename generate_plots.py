"""
generate_plots.py — Auto-generate all report graphs
=====================================================
Run this after all training is complete to generate:
  1. Cumulative reward curves — all 4 algorithms (subplots)
  2. DQN objective / loss curves
  3. Policy entropy curves — PPO, A2C, REINFORCE
  4. Convergence comparison bar chart
  5. Best episode reward comparison
  6. Generalization test results

Usage:
    python generate_plots.py

All plots saved to: plots/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Setup ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor":  "#0F141E",
    "axes.facecolor":    "#161E2D",
    "axes.edgecolor":    "#3A4A6A",
    "axes.labelcolor":   "#C8D0E0",
    "axes.titlecolor":   "#E0E8F0",
    "xtick.color":       "#8090AA",
    "ytick.color":       "#8090AA",
    "text.color":        "#C8D0E0",
    "grid.color":        "#2A3A5A",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1A2640",
    "legend.edgecolor":  "#3A4A6A",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

ALGO_COLORS = {
    "DQN":       "#5090F0",
    "PPO":       "#B064DC",
    "A2C":       "#F0A020",
    "REINFORCE": "#40C880",
}

ALGO_DIRS = {
    "DQN":       os.path.join("logs", "dqn"),
    "PPO":       os.path.join("logs", "ppo"),
    "A2C":       os.path.join("logs", "a2c"),
    "REINFORCE": os.path.join("logs", "reinforce"),
}


# ── Helper: load reward arrays ────────────────────────────────────────────────

def load_rewards(algo: str) -> dict:
    """Load all episode_rewards.npy files for an algorithm."""
    log_dir = ALGO_DIRS[algo]
    rewards  = {}
    if not os.path.exists(log_dir):
        return rewards
    for exp_folder in sorted(os.listdir(log_dir)):
        npy_path = os.path.join(log_dir, exp_folder, "episode_rewards.npy")
        if os.path.exists(npy_path):
            rewards[exp_folder] = np.load(npy_path)
    return rewards


def smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling mean smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_csv(algo: str) -> pd.DataFrame:
    """Load results CSV for an algorithm."""
    csv_map = {
        "DQN":       os.path.join("logs", "dqn", "dqn_results.csv"),
        "PPO":       os.path.join("logs", "ppo", "ppo_results.csv"),
        "A2C":       os.path.join("logs", "a2c", "a2c_results.csv"),
        "REINFORCE": os.path.join("logs", "reinforce", "reinforce_results.csv"),
    }
    path = csv_map.get(algo, "")
    if os.path.exists(path):
        return pd.read_csv(path, encoding='latin-1')
    return pd.DataFrame()


# ── Plot 1: Cumulative reward curves — 4 subplots ────────────────────────────

def plot_cumulative_rewards():
    print("  Generating Plot 1: Cumulative reward curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Episode Reward Curves — All Algorithms (Smoothed)\nRwanda Civic Dispatch RL",
                 fontsize=14, fontweight="bold", y=0.98)

    algos = ["DQN", "PPO", "A2C", "REINFORCE"]
    for ax, algo in zip(axes.flat, algos):
        rewards_dict = load_rewards(algo)
        color = ALGO_COLORS[algo]

        if not rewards_dict:
            ax.text(0.5, 0.5, f"{algo}\nNo data found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(algo)
            continue

        all_smoothed = []
        best_raw_reward = -np.inf
        best_exp_name   = ""

        for exp_name, rewards in rewards_dict.items():
            if len(rewards) < 10:
                continue
            # Smooth the per-episode reward — shows learning progress clearly
            smoothed = smooth(rewards, window=30)
            ax.plot(smoothed, color=color, alpha=0.25, linewidth=0.8)
            all_smoothed.append((exp_name, smoothed, rewards))

            # Track best experiment by max smoothed reward reached
            if np.max(smoothed) > best_raw_reward:
                best_raw_reward = np.max(smoothed)
                best_exp_name   = exp_name

        if all_smoothed:
            # Mean curve across all experiments
            max_len    = max(len(s) for _, s, _ in all_smoothed)
            padded     = [np.pad(s, (0, max_len - len(s)), constant_values=s[-1])
                          for _, s, _ in all_smoothed]
            mean_curve = np.mean(padded, axis=0)
            ax.plot(mean_curve, color=color, linewidth=2.5,
                    label=f"{algo} mean", zorder=5)

            # Highlight best experiment
            for exp_name, smoothed, _ in all_smoothed:
                if exp_name == best_exp_name:
                    ax.plot(smoothed, color="white", linewidth=1.8,
                            linestyle="--", label=f"Best: {exp_name}", zorder=6)
                    break

            # Mark the zero line — crossing it means agent is profitable
            ax.axhline(y=0, color="red", linewidth=1.0,
                       linestyle=":", alpha=0.7, label="Break-even (0)")

        ax.set_title(f"{algo}", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (smoothed)")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(PLOTS_DIR, "01_cumulative_rewards.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Plot 2: Episode reward comparison — best per algo ────────────────────────

def plot_best_episode_comparison():
    print("  Generating Plot 2: Best episode reward comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Best Episode Reward per Experiment — All Algorithms",
                 fontsize=13, fontweight="bold")

    algos  = ["DQN", "PPO", "A2C", "REINFORCE"]
    offset = 0
    ticks, tick_labels = [], []

    for algo in algos:
        df = load_csv(algo)
        if df.empty:
            continue

        reward_col = "Best Episode Reward" if "Best Episode Reward" in df.columns else df.columns[-2]
        values = df[reward_col].values
        color  = ALGO_COLORS[algo]

        x_pos = np.arange(len(values)) + offset
        bars  = ax.bar(x_pos, values, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.5, label=algo)

        # Value labels on bars
        for bar, val in zip(bars, values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (5 if y >= 0 else -15),
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=7, color="white")

        ticks      += list(x_pos)
        tick_labels += [f"E{i+1}" for i in range(len(values))]
        offset     += len(values) + 2

        # Algo label
        mid = x_pos[len(x_pos) // 2]
        ax.text(mid, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 10,
                algo, ha="center", fontsize=10, color=color, fontweight="bold")

    ax.axhline(y=0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_ylabel("Best Episode Reward")
    ax.set_xlabel("Experiment")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_best_episode_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Plot 3: Convergence episodes comparison ───────────────────────────────────

def plot_convergence():
    print("  Generating Plot 3: Convergence comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Episodes to Convergence — All Algorithms",
                 fontsize=13, fontweight="bold")

    algos = ["DQN", "PPO", "A2C", "REINFORCE"]

    for i, algo in enumerate(algos):
        df = load_csv(algo)
        if df.empty:
            continue

        conv_col = "Convergence Episode" if "Convergence Episode" in df.columns else df.columns[-1]
        values   = df[conv_col].values
        color    = ALGO_COLORS[algo]

        x = np.arange(len(values)) + i * 12
        ax.bar(x, values, color=color, alpha=0.8,
               edgecolor="white", linewidth=0.5, label=algo)

        # Mean line
        mean_val = np.mean(values)
        ax.hlines(mean_val, x[0] - 0.5, x[-1] + 0.5,
                  colors=color, linewidth=2, linestyle="--",
                  label=f"{algo} mean: {mean_val:.0f}")

    ax.set_ylabel("Episodes to Convergence")
    ax.set_xlabel("Experiments")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_convergence_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Plot 4: Training stability — rolling mean per algo ────────────────────────

def plot_training_stability():
    print("  Generating Plot 4: Training stability...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Stability — Rolling Mean Reward (window=50)\nRwanda Civic Dispatch RL",
                 fontsize=13, fontweight="bold")

    algos = ["DQN", "PPO", "A2C", "REINFORCE"]
    for ax, algo in zip(axes.flat, algos):
        rewards_dict = load_rewards(algo)
        color = ALGO_COLORS[algo]

        if not rewards_dict:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        for exp_name, rewards in rewards_dict.items():
            if len(rewards) < 50:
                continue
            rolled = smooth(rewards, window=50)
            ax.plot(rolled, color=color, alpha=0.3, linewidth=1)

        # Overall mean stability
        all_r = [smooth(r, 50) for r in rewards_dict.values() if len(r) >= 50]
        if all_r:
            min_len  = min(len(r) for r in all_r)
            trimmed  = [r[:min_len] for r in all_r]
            mean_r   = np.mean(trimmed, axis=0)
            std_r    = np.std(trimmed, axis=0)
            x        = np.arange(min_len)

            ax.plot(mean_r, color=color, linewidth=2.5, label="Mean")
            ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                            color=color, alpha=0.15, label="±1 std")

        ax.axhline(y=0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(f"{algo} — Training Stability", color=color, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rolling Mean Reward")
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(PLOTS_DIR, "04_training_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Plot 5: Algorithm head-to-head — final mean reward ───────────────────────

def plot_algorithm_comparison():
    print("  Generating Plot 5: Algorithm head-to-head...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Algorithm Head-to-Head Comparison — Rwanda Civic Dispatch RL",
                 fontsize=13, fontweight="bold")

    algos = ["DQN", "PPO", "A2C", "REINFORCE"]

    # Left: Final mean reward per algo (box plot style)
    ax = axes[0]
    all_finals = []
    labels     = []
    colors     = []

    for algo in algos:
        df = load_csv(algo)
        if df.empty:
            continue
        col = "Final Mean Reward" if "Final Mean Reward" in df.columns else df.columns[-3]
        all_finals.append(df[col].values)
        labels.append(algo)
        colors.append(ALGO_COLORS[algo])

    bp = ax.boxplot(all_finals, patch_artist=True, labels=labels,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Final Mean Reward Distribution")
    ax.set_ylabel("Final Mean Reward (last 20 eps)")
    ax.grid(True, axis="y")

    # Right: Best episode per algo
    ax = axes[1]
    best_vals  = []
    best_label = []

    for algo in algos:
        df = load_csv(algo)
        if df.empty:
            continue
        col = "Best Episode Reward" if "Best Episode Reward" in df.columns else df.columns[-2]
        best_vals.append(df[col].max())
        best_label.append(algo)

    bar_colors = [ALGO_COLORS[a] for a in best_label]
    bars = ax.bar(best_label, best_vals, color=bar_colors,
                  edgecolor="white", linewidth=0.8, alpha=0.85)

    for bar, val in zip(bars, best_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{val:.1f}", ha="center", fontsize=10,
                color="white", fontweight="bold")

    ax.set_title("Best Single Episode Reward per Algorithm")
    ax.set_ylabel("Best Episode Reward")
    ax.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(PLOTS_DIR, "05_algorithm_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Plot 6: Generalization test ───────────────────────────────────────────────

def plot_generalization():
    """
    Test trained models on unseen initial states (different seeds).
    Uses best model per algorithm if available, otherwise simulates.
    """
    print("  Generating Plot 6: Generalization test...")

    from environment.custom_env import CivicIssueEnv

    # Try to load best models for generalization test
    results = {}
    test_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    algos_to_test = []

    # Check which SB3 models are available
    for algo, model_dir in [("DQN",  "models/dqn"),
                             ("PPO",  "models/ppo"),
                             ("A2C",  "models/a2c")]:
        if not os.path.exists(model_dir):
            continue
        # Find best model
        for exp_folder in sorted(os.listdir(model_dir)):
            best_zip = os.path.join(model_dir, exp_folder, "best_model.zip")
            if os.path.exists(best_zip):
                algos_to_test.append((algo, best_zip[:-4]))
                break

    # REINFORCE
    reinforce_dir = "models/reinforce"
    if os.path.exists(reinforce_dir):
        for exp_folder in sorted(os.listdir(reinforce_dir)):
            pt_path = os.path.join(reinforce_dir, exp_folder, "policy.pt")
            if os.path.exists(pt_path):
                algos_to_test.append(("REINFORCE", pt_path))
                break

    if not algos_to_test:
        print("    No trained models found — generating placeholder plot")
        _plot_generalization_placeholder()
        return

    env = CivicIssueEnv()

    for algo_name, model_path in algos_to_test:
        seed_rewards = []

        try:
            if algo_name == "DQN":
                from stable_baselines3 import DQN
                model = DQN.load(model_path)
            elif algo_name == "PPO":
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
            elif algo_name == "A2C":
                from stable_baselines3 import A2C
                model = A2C.load(model_path)
            elif algo_name == "REINFORCE":
                import torch
                import torch.nn as nn
                from training.reinforce_training import PolicyNetwork
                policy = PolicyNetwork(60, 12, 128)
                policy.load_state_dict(torch.load(model_path,
                                       map_location="cpu"))
                policy.eval()

            for seed in test_seeds:
                obs, _ = env.reset(seed=seed)
                total_r = 0
                done    = False
                while not done:
                    if algo_name == "REINFORCE":
                        import torch
                        with torch.no_grad():
                            probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                            action = probs.argmax().item()
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(int(action))
                    total_r += r
                    done = term or trunc
                seed_rewards.append(total_r)

            results[algo_name] = seed_rewards
            print(f"    {algo_name}: mean={np.mean(seed_rewards):.1f} "
                  f"std={np.std(seed_rewards):.1f}")

        except Exception as e:
            print(f"    {algo_name}: skipped ({e})")

    env.close()

    if not results:
        _plot_generalization_placeholder()
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Generalization Test — 10 Unseen Initial States",
                 fontsize=13, fontweight="bold")

    # Left: line plot per seed
    ax = axes[0]
    for algo, rewards in results.items():
        ax.plot(test_seeds, rewards, marker="o", color=ALGO_COLORS[algo],
                linewidth=2, markersize=6, label=algo)
    ax.axhline(y=0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Test Seed")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward per Unseen Initial State")
    ax.legend()
    ax.grid(True)

    # Right: mean ± std
    ax = axes[1]
    algo_names = list(results.keys())
    means = [np.mean(results[a]) for a in algo_names]
    stds  = [np.std(results[a])  for a in algo_names]
    colors = [ALGO_COLORS[a] for a in algo_names]

    bars = ax.bar(algo_names, means, yerr=stds, color=colors,
                  alpha=0.8, edgecolor="white", capsize=6,
                  error_kw={"ecolor": "white", "linewidth": 1.5})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.1f}", ha="center", fontsize=10,
                color="white", fontweight="bold")

    ax.axhline(y=0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Mean ± Std Across 10 Seeds")
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(PLOTS_DIR, "06_generalization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def _plot_generalization_placeholder():
    """Placeholder when no trained models are found."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Generalization Test — Placeholder", fontsize=13)
    ax.text(0.5, 0.5, "Run training first, then re-run generate_plots.py",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    path = os.path.join(PLOTS_DIR, "06_generalization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved placeholder: {path}")


# ── Plot 7: DQN hyperparameter sensitivity ────────────────────────────────────

def plot_dqn_sensitivity():
    print("  Generating Plot 7: DQN hyperparameter sensitivity...")
    df = load_csv("DQN")
    if df.empty:
        print("    No DQN CSV found — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("DQN Hyperparameter Sensitivity — Rwanda Civic Dispatch",
                 fontsize=13, fontweight="bold")

    color = ALGO_COLORS["DQN"]
    reward_col = "Best Episode Reward" if "Best Episode Reward" in df.columns else df.columns[-2]

    # LR vs Best Reward
    ax = axes[0]
    ax.scatter(df["Learning Rate"], df[reward_col],
               color=color, s=80, zorder=5)
    for _, row in df.iterrows():
        ax.annotate(row["Experiment"].replace("Exp0", "E").replace("Exp", "E"),
                    (row["Learning Rate"], row[reward_col]),
                    fontsize=7, color="white", xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Best Episode Reward")
    ax.set_title("LR vs Best Reward")
    ax.grid(True)

    # Gamma vs Best Reward
    ax = axes[1]
    ax.scatter(df["Gamma"], df[reward_col],
               color=color, s=80, zorder=5)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Best Episode Reward")
    ax.set_title("Gamma vs Best Reward")
    ax.grid(True)

    # Buffer Size vs Best Reward
    ax = axes[2]
    if "Buffer Size" in df.columns:
        ax.scatter(df["Buffer Size"], df[reward_col],
                   color=color, s=80, zorder=5)
        ax.set_xlabel("Buffer Size")
        ax.set_title("Buffer Size vs Best Reward")
    else:
        ax.text(0.5, 0.5, "Buffer Size column not found",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Best Episode Reward")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(PLOTS_DIR, "07_dqn_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  GENERATING ALL REPORT PLOTS")
    print("  Rwanda Civic Dispatch RL")
    print("="*60 + "\n")

    plot_cumulative_rewards()
    plot_best_episode_comparison()
    plot_convergence()
    plot_training_stability()
    plot_algorithm_comparison()
    plot_generalization()
    plot_dqn_sensitivity()

    print(f"\n{'='*60}")
    print(f"  ✅ All plots saved to: {PLOTS_DIR}/")
    print(f"\n  Files generated:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"    {f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()