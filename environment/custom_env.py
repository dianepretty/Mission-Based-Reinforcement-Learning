"""
CivicIssueEnv — Rwanda Civic Issue Reporting RL Environment
============================================================
Mission: tackle citizen disengagement and lack of government accountability
by training an agent to optimally dispatch citizen reports across
government institutions.

Simplified Environment Design
------------------------------
  Agent     : The dispatch AI inside the civic platform
  State     : 10 pending reports + 4 institution states + 4 global metrics
  Actions   : 5 discrete — dispatch to NLA/WASAC/REG/MININFRA or wait
  Reward    : urgency-weighted resolution speed + trust delta − neglect penalty
  Terminal  : 200 steps only (no early collapse — episode always runs full month)

Observation vector (60 floats)
-------------------------------
  [0:40]  — 10 reports × 4 features each:
              urgency (0-5), category (0-3), days_waiting (0-30), evidence (0-1)
  [40:56] — 4 institutions × 4 features each:
              capacity (0-1), resolution_rate (0-1), workload (0-5), response_time (0-10)
  [56:60] — global: trust (0-100), resolved_today (0-20), critical_pending (0-10), step_norm (0-1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

INSTITUTIONS     = ["NLA", "WASAC", "REG", "MININFRA"]
CATEGORY_NAMES   = ["Land", "Water", "Electricity", "Roads"]
CATEGORY_TO_INST = [0, 1, 2, 3]

NUM_REPORTS      = 10
NUM_INSTITUTIONS = 4
MAX_STEPS        = 200
INITIAL_TRUST    = 90.0
NUM_ACTIONS      = 5    # 0-3: dispatch to institution, 4: wait
OBS_SIZE         = NUM_REPORTS * 4 + NUM_INSTITUTIONS * 4 + 4  # = 60


class CivicIssueEnv(gym.Env):
    """
    Custom Gymnasium environment for civic issue dispatch optimisation.
    Compatible with Stable Baselines 3 (DQN, PPO, A2C).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: 60 floats normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(OBS_SIZE,  dtype=np.float32),
            high=np.ones(OBS_SIZE,  dtype=np.float32),
            dtype=np.float32,
        )

        self.reports            = []
        self.institutions       = []
        self.trust              = INITIAL_TRUST
        self.step_count         = 0
        self.resolved_today     = 0
        self.critical_pending   = 0
        self.total_reward       = 0.0
        self.last_action_label  = "—"
        self.event_log          = []
        self._renderer          = None

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Start with 4-6 reports
        self.reports = []
        for _ in range(rng.integers(4, 7)):
            self.reports.append(self._spawn_report(rng))
        while len(self.reports) < NUM_REPORTS:
            self.reports.append(self._empty_report())

        # Initialise institutions
        self.institutions = []
        for i in range(NUM_INSTITUTIONS):
            self.institutions.append({
                "name":            INSTITUTIONS[i],
                "capacity":        rng.uniform(0.6, 1.0),
                "resolution_rate": rng.uniform(0.65, 0.95),
                "workload":        int(rng.integers(0, 2)),
                "response_time":   rng.uniform(1.0, 4.0),
            })

        self.trust              = INITIAL_TRUST
        self.step_count         = 0
        self.resolved_today     = 0
        self.critical_pending   = 0
        self.total_reward       = 0.0
        self.last_action_label  = "System initialised"
        self.event_log          = ["[ Civic dispatch system online ]"]

        return self._get_obs(), self._get_info()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: int):
        self.step_count     += 1
        self.resolved_today  = 0
        reward               = 0.0

        # ── Decode action ─────────────────────────────────────────
        if action < 4:
            # Dispatch the most urgent active report to institution [action]
            reward += self._dispatch_urgent(inst_idx=action)
        else:
            # Wait — small penalty to discourage idle behavior
            reward -= 1.0
            self.last_action_label = "⏸ Agent waited"
            self._log("⏸ Agent waited this step")

        # ── Environment dynamics ──────────────────────────────────
        reward += self._age_reports()
        self._maybe_spawn_report()
        self._update_institutions()
        trust_delta = self._update_trust()
        reward += trust_delta * 0.3

        # Small penalty per critical report still pending
        self.critical_pending = sum(
            1 for r in self.reports if r["active"] and r["urgency"] >= 4
        )
        reward -= self.critical_pending * 0.1

        # Always run full 200 steps — no early termination
        terminated = False
        truncated  = self.step_count >= MAX_STEPS

        self.total_reward += reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _dispatch_urgent(self, inst_idx: int) -> float:
        """Dispatch the most urgent active report to institution inst_idx."""
        # Find the most urgent active report
        candidates = [(i, r) for i, r in enumerate(self.reports) if r["active"]]

        if not candidates:
            self._log(f"✗ No active reports to dispatch")
            return -2.0

        # Pick highest urgency report
        slot, report = max(candidates, key=lambda x: (x[1]["urgency"], x[1]["days_waiting"]))
        inst = self.institutions[inst_idx]

        # Correct institution match bonus
        correct     = (CATEGORY_TO_INST[report["category"]] == inst_idx)
        match_bonus = 8.0 if correct else -1.0

        # Resolution reward: urgency × speed
        speed  = max(0.2, 1.0 - (report["days_waiting"] / 30.0))
        r      = (report["urgency"] * 8.0) * speed + match_bonus

        # Evidence quality multiplier
        r *= (0.8 + 0.2 * report["evidence"])

        # Institution capacity penalty if overloaded
        if inst["capacity"] < 0.2:
            r *= 0.5
            self._log(f"⚠ {inst['name']} overloaded")

        # Update institution
        inst["workload"] = min(5, inst["workload"] + 1)
        inst["capacity"] = max(0.0, inst["capacity"] - 0.08)

        # Resolve report
        self.resolved_today    += 1
        self.reports[slot]      = self._empty_report()

        label = (f"→ Dispatched [{CATEGORY_NAMES[report['category']]}] "
                 f"urgency {report['urgency']} to {inst['name']} | +{r:.1f}")
        self.last_action_label = label
        self._log(label)
        return r

    # ── Dynamics ─────────────────────────────────────────────────────────────

    def _age_reports(self) -> float:
        """Age all active reports by 1 day."""
        penalty = 0.0
        for report in self.reports:
            if not report["active"]:
                continue
            report["days_waiting"] += 1
            # Mild penalty for neglecting critical reports
            if report["urgency"] >= 4 and report["days_waiting"] > 10:
                penalty -= 0.5
            # Urgency creep after long wait
            if report["days_waiting"] > 12 and report["urgency"] < 5:
                if random.random() < 0.08:
                    report["urgency"] += 1
        return penalty

    def _maybe_spawn_report(self):
        """15% chance each step a new report arrives."""
        if random.random() < 0.15:
            empty = [i for i, r in enumerate(self.reports) if not r["active"]]
            if empty:
                slot = random.choice(empty)
                self.reports[slot] = self._spawn_report(self.np_random)
                r = self.reports[slot]
                self._log(f"+ New [{CATEGORY_NAMES[r['category']]}] urgency {r['urgency']}")

    def _update_institutions(self):
        """Institutions recover capacity each step."""
        for inst in self.institutions:
            inst["capacity"]      = min(1.0, inst["capacity"] + 0.06)
            inst["workload"]      = max(0,   inst["workload"] - 1)
            inst["response_time"] = max(1.0, inst["response_time"] - 0.05)

    def _update_trust(self) -> float:
        """Trust evolves based on resolutions vs neglect."""
        delta  = 0.0
        delta += self.resolved_today * 3.0
        delta -= self.critical_pending * 0.5
        avg_wait = np.mean(
            [r["days_waiting"] for r in self.reports if r["active"]] or [0]
        )
        delta -= avg_wait * 0.05
        self.trust = float(np.clip(self.trust + delta, 0.0, 100.0))
        return delta

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _spawn_report(self, rng) -> dict:
        return {
            "active":       True,
            "urgency":      int(rng.integers(1, 6)),
            "category":     int(rng.integers(0, NUM_INSTITUTIONS)),
            "days_waiting": int(rng.integers(0, 4)),
            "evidence":     float(rng.uniform(0.2, 1.0)),
        }

    def _empty_report(self) -> dict:
        return {
            "active": False, "urgency": 0,
            "category": 0, "days_waiting": 0, "evidence": 0.0,
        }

    def _log(self, msg: str):
        entry = f"Step {self.step_count:03d} | {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > 6:
            self.event_log.pop(0)

    # ── Observation & Info ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        ptr = 0
        for r in self.reports:
            obs[ptr]     = r["urgency"] / 5.0
            obs[ptr + 1] = r["category"] / (NUM_INSTITUTIONS - 1)
            obs[ptr + 2] = min(r["days_waiting"], 30) / 30.0
            obs[ptr + 3] = r["evidence"]
            ptr += 4
        for inst in self.institutions:
            obs[ptr]     = inst["capacity"]
            obs[ptr + 1] = inst["resolution_rate"]
            obs[ptr + 2] = inst["workload"] / 5.0
            obs[ptr + 3] = inst["response_time"] / 10.0
            ptr += 4
        obs[ptr]     = self.trust / 100.0
        obs[ptr + 1] = min(self.resolved_today, 20) / 20.0
        obs[ptr + 2] = min(self.critical_pending, 10) / 10.0
        obs[ptr + 3] = self.step_count / MAX_STEPS
        return obs

    def _get_info(self) -> dict:
        return {
            "trust":            self.trust,
            "step":             self.step_count,
            "resolved_today":   self.resolved_today,
            "critical_pending": self.critical_pending,
            "total_reward":     self.total_reward,
            "last_action":      self.last_action_label,
            "event_log":        self.event_log.copy(),
            "reports":          self.reports.copy(),
            "institutions":     self.institutions.copy(),
        }

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return
        if self._renderer is None:
            from environment.rendering import CivicRenderer
            self._renderer = CivicRenderer(width=1100, height=680)
        return self._renderer.render(self._get_info())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = CivicIssueEnv()
    obs, info = env.reset(seed=42)
    print(f"Obs shape : {obs.shape}")
    print(f"Actions   : {env.action_space.n}")
    print(f"Trust     : {info['trust']}")
    print(f"Reports   : {sum(1 for r in info['reports'] if r['active'])} active")

    total_r = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_r += reward
        print(f"  Step {step+1:02d} | action={action} | "
              f"reward={reward:+.2f} | trust={info['trust']:.1f} | "
              f"{info['last_action'][:50]}")
    print(f"\nTotal reward: {total_r:.2f}")
    env.close()
    print("✓ Environment OK")