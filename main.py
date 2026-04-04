"""
main.py — Rwanda Civic Dispatch RL
====================================
Runs the best trained agent with full pygame dashboard.

Usage:
    python main.py
    python main.py --algo ppo
    python main.py --algo dqn
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
    search_algos = [algo.upper()] if algo else ["PPO", "DQN", "A2C"]
    for algo_name in search_algos:
        model_dir = os.path.join("models", algo_name.lower())
        if not os.path.exists(model_dir):
            continue
        for exp_folder in sorted(os.listdir(model_dir)):
            exp_path = os.path.join(model_dir, exp_folder)
            if not os.path.isdir(exp_path):
                continue
            for fname in ["best_model.zip", "final_model.zip"]:
                full = os.path.join(exp_path, fname)
                if os.path.exists(full):
                    return full[:-4], algo_name
        for fname in os.listdir(model_dir):
            if fname.endswith(".zip"):
                return os.path.join(model_dir, fname[:-4]), algo_name
    return None, None


def load_model(model_path, algo_name):
    if algo_name.upper() == "DQN":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo_name.upper() == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo_name.upper() == "A2C":
        from stable_baselines3 import A2C
        return A2C.load(model_path)


def run(algo=None, max_episodes=None):
    print("\n" + "="*60)
    print("  Rwanda Civic Dispatch — RL Agent Dashboard")
    print("="*60)

    model_path, algo_name = find_best_model(algo)

    if model_path is None:
        print("\n No trained model found. Searching:")
        for folder in ["models/dqn", "models/ppo", "models/a2c"]:
            if os.path.exists(folder):
                print(f"  {folder}/ -> {os.listdir(folder)}")
        return

    print(f"\n  Algorithm  : {algo_name}")
    print(f"  Model      : {model_path}.zip")

    try:
        model = load_model(model_path, algo_name)
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    env      = CivicIssueEnv()
    renderer = CivicRenderer(width=1100, height=680)

    print(f"  Dashboard open — close window to stop\n")

    episode = 0
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
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_reward += reward
                step += 1
                done = terminated or truncated

                renderer.render(info)
                pygame.time.wait(250)

            total_rewards.append(episode_reward)
            status = "COLLAPSE" if terminated else "complete"
            print(f"  Episode {episode} {status} | Steps: {step} | "
                  f"Reward: {episode_reward:+.1f} | Trust: {info['trust']:.0f}%")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        renderer.close()
        env.close()

    if total_rewards:
        print(f"\n  Mean: {np.mean(total_rewards):+.2f} | "
              f"Best: {np.max(total_rewards):+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    run(algo=args.algo, max_episodes=args.episodes)