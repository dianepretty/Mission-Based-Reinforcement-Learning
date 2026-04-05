"""
main.py — Rwanda Civic Dispatch RL
====================================
Runs the best trained agent with full pygame dashboard.

Usage:
    python main.py
    python main.py --algo dqn
    python main.py --algo ppo
    python main.py --algo a2c
"""

import os
import sys
import time
import argparse
import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.custom_env import CivicIssueEnv
from environment.rendering import CivicRenderer


def find_best_model(algo=None):
    """Find best model — checks best_model/ folder first."""

    # Priority 1: best_model folder (manually placed best model)
    best_dir = os.path.join("models", "best_model")
    if os.path.exists(best_dir):
        for fname in ["best_model.zip", "final_model.zip"]:
            full = os.path.join(best_dir, fname)
            if os.path.exists(full):
                algo_name = algo.upper() if algo else "DQN"
                print(f"  Found model in best_model/ folder")
                return full[:-4], algo_name

    # Priority 2: search by algorithm folder
    search_algos = [algo.upper()] if algo else ["DQN", "PPO", "A2C"]

    for algo_name in search_algos:
        model_dir = os.path.join("models", algo_name.lower())
        if not os.path.exists(model_dir):
            continue

        # Check preferred experiments first
        priority = ["Exp06_LargeBuffer", "Exp10_Balanced", "Exp08_LargeBatch"]
        all_folders = sorted(os.listdir(model_dir))
        ordered = [f for f in priority if f in all_folders] + \
                  [f for f in all_folders if f not in priority]

        for exp_folder in ordered:
            exp_path = os.path.join(model_dir, exp_folder)
            if not os.path.isdir(exp_path):
                continue
            for fname in ["best_model.zip", "final_model.zip"]:
                full = os.path.join(exp_path, fname)
                if os.path.exists(full):
                    print(f"  Found model in {exp_path}/")
                    return full[:-4], algo_name

    return None, None


def load_model(model_path, algo_name):
    """Load SB3 model by algorithm name."""
    if algo_name.upper() == "DQN":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo_name.upper() == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo_name.upper() == "A2C":
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run(algo=None, max_episodes=None):
    print("\n" + "="*60)
    print("  Rwanda Civic Dispatch — RL Agent Dashboard")
    print("="*60)

    # ── Find model ────────────────────────────────────────────────
    model_path, algo_name = find_best_model(algo)

    if model_path is None:
        print("\n  No trained model found!")
        print("  Make sure models exist in models/best_model/ or models/dqn/")
        return

    print(f"\n  Algorithm  : {algo_name}")
    print(f"  Model      : {model_path}.zip")

    try:
        model = load_model(model_path, algo_name)
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  Failed to load: {e}")
        return

    # ── Setup ─────────────────────────────────────────────────────
    env      = CivicIssueEnv()
    renderer = CivicRenderer(width=1100, height=680)

    print(f"\n  Dashboard open — close window to stop\n")

    episode       = 0
    total_rewards = []

    try:
        while True:
            episode += 1
            if max_episodes and episode > max_episodes:
                break

            obs, info = env.reset()
            episode_reward = 0
            step = 0
            done = False

            print(f"  Episode {episode} starting...")

            while not done:
                # Handle window close
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                # Agent picks best action
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_reward += reward
                step += 1
                done = terminated or truncated

                # Render dashboard
                renderer.render(info)
                pygame.time.wait(800)

            total_rewards.append(episode_reward)
            status = "COLLAPSE" if terminated else "complete"
            print(f"  Episode {episode} {status} | "
                  f"Steps: {step} | "
                  f"Reward: {episode_reward:+.1f} | "
                  f"Trust: {info['trust']:.0f}%")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        renderer.close()
        env.close()

    if total_rewards:
        print(f"\n{'='*60}")
        print(f"  Session Summary ({len(total_rewards)} episodes)")
        print(f"  Mean reward  : {np.mean(total_rewards):+.2f}")
        print(f"  Best episode : {np.max(total_rewards):+.2f}")
        print(f"  Worst episode: {np.min(total_rewards):+.2f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=None,
                        choices=["dqn", "ppo", "a2c", "DQN", "PPO", "A2C"])
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    run(algo=args.algo, max_episodes=args.episodes)