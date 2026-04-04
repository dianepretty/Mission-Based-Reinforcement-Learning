"""
demo.py — Random Agent Visualization
======================================
Demonstrates the custom environment with a RANDOM agent
(no trained model required). This satisfies the assignment
requirement to show the agent taking random actions in the
environment before any training.

This file shows:
  - The environment's action and observation spaces
  - All possible agent behaviors (random)
  - The full pygame dashboard visualization
  - Episode resets and terminal conditions

Usage:
    python demo.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.custom_env import CivicIssueEnv
from environment.rendering import CivicRenderer

import pygame


def print_env_summary(env: CivicIssueEnv):
    """Print environment details to terminal for the video demo."""
    print("\n" + "="*60)
    print("  ENVIRONMENT SUMMARY — Rwanda Civic Dispatch RL")
    print("="*60)
    print(f"\n  Observation space : Box({env.observation_space.shape[0]},)")
    print(f"  Action space      : Discrete({env.action_space.n})")
    print(f"\n  Actions:")
    print(f"    0-3  : Dispatch slot-0 report to NLA/WASAC/REG/MININFRA")
    print(f"    4    : Escalate most urgent report")
    print(f"    5    : Request more evidence")
    print(f"    6    : Defer lowest-urgency report")
    print(f"    7    : Close report as duplicate")
    print(f"    8-11 : Dispatch slot-1 report to NLA/WASAC/REG/MININFRA")
    print(f"\n  Reward structure:")
    print(f"    + urgency × speed    : resolving reports quickly")
    print(f"    + trust delta × 0.5  : citizen trust growing")
    print(f"    - neglect penalty    : critical reports ignored")
    print(f"    - 50                 : trust collapse (< 20%)")
    print(f"\n  Terminal conditions:")
    print(f"    truncated  : 200 steps completed (1 simulated month)")
    print(f"    terminated : trust drops below 20% (system collapse)")
    print(f"\n  RANDOM AGENT — no model, sampling action space uniformly")
    print("="*60 + "\n")


def run_demo(num_episodes: int = 5, steps_per_episode: int = 200):
    """Run random agent for visualization."""

    env      = CivicIssueEnv()
    renderer = CivicRenderer(width=1100, height=680)

    print_env_summary(env)
    print("  Starting random agent demo — close window to stop\n")

    all_rewards = []
    all_trusts  = []

    try:
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset(seed=episode)
            episode_reward = 0
            step_count     = 0

            print(f"  Episode {episode}/{num_episodes} | Random agent starting...")

            while True:
                # Handle pygame quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                # RANDOM ACTION — uniformly sampled from action space
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count     += 1

                # Render
                renderer.render(info)

                # Slow enough to see what's happening
                pygame.time.wait(200)

                if terminated or truncated:
                    all_rewards.append(episode_reward)
                    all_trusts.append(info["trust"])

                    status = "COLLAPSE ⚠" if terminated else "complete ✓"
                    print(f"  Episode {episode} {status} | "
                          f"Steps: {step_count:3d} | "
                          f"Reward: {episode_reward:+7.1f} | "
                          f"Trust: {info['trust']:.0f}%")

                    # Pause briefly between episodes
                    time.sleep(1.5)
                    break

    except KeyboardInterrupt:
        print("\n  Demo stopped.")

    finally:
        renderer.close()
        env.close()

    # ── Terminal summary ──────────────────────────────────────────
    if all_rewards:
        print(f"\n{'='*60}")
        print(f"  RANDOM AGENT SUMMARY ({len(all_rewards)} episodes)")
        print(f"  Mean reward : {np.mean(all_rewards):+.2f}")
        print(f"  Mean trust  : {np.mean(all_trusts):.1f}%")
        print(f"  Best reward : {np.max(all_rewards):+.2f}")
        print(f"  Worst reward: {np.min(all_rewards):+.2f}")
        print(f"\n  Note: A trained agent will significantly outperform")
        print(f"  these random baseline numbers.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    run_demo(num_episodes=5)del train_best.py