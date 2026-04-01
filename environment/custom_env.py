"""
CivicIssueEnv — Rwanda Civic Issue Reporting RL Environment
============================================================
Mission: tackle citizen disengagement and lack of government accountability
by training an agent to optimally dispatch, escalate, and manage citizen
reports across government institutions.

Environment Design
------------------
  Agent     : The dispatch AI inside the civic platform
  State     : 10 pending reports + 4 institution states + 4 global metrics
  Actions   : 12 discrete — dispatch/escalate/defer/close per report slot,
              plus request-evidence and wait
  Reward    : urgency-weighted resolution speed + trust delta − neglect penalty
  Terminal  : 200 steps (one simulated month) OR trust collapse below 20%

Observation vector (56 floats)
-------------------------------
  [0:40]  — 10 reports × 4 features each:
              urgency (0–5), category (0–3), days_waiting (0–30), evidence (0–1)
  [40:56] — 4 institutions × 4 features each:
              capacity (0–1), resolution_rate (0–1), workload (0–5), response_time (0–10)
  [56:60] — global: trust (0–100), resolved_today (0–20), critical_pending (0–10), step_norm (0–1)

Total: 60 floats
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

INSTITUTIONS = ["NLA", "WASAC", "REG", "MININFRA"]

# Which issue categories map to which institutions
CATEGORY_NAMES    = ["Land", "Water", "Electricity", "Roads"]
CATEGORY_TO_INST  = [0, 1, 2, 3]   # Land→NLA, Water→WASAC, Elec→REG, Roads→MININFRA

NUM_REPORTS       = 10
NUM_INSTITUTIONS  = 4
MAX_STEPS         = 200
TRUST_COLLAPSE    = 20.0     # episode ends if trust drops below this
INITIAL_TRUST     = 70.0

# Action ids
# 0–3   : dispatch report in slot 0 to institution 0-3
# 4     : escalate top-urgency pending report
# 5     : request more evidence from citizen (top pending)
# 6     : defer lowest-urgency report (buy time)
# 7     : close top report as duplicate/invalid
# 8–11  : dispatch report in slot 1 to institution 0-3
# We keep it at 12 discrete actions — enough complexity to differentiate algos
NUM_ACTIONS = 12

OBS_SIZE = NUM_REPORTS * 4 + NUM_INSTITUTIONS * 4 + 4   # = 60


class CivicIssueEnv(gym.Env):
    """
    Custom Gymnasium environment for civic issue dispatch optimisation.
    Implements the full Gymnasium API: reset(), step(), render().
    Compatible with Stable Baselines 3 (DQN, PPO, A2C).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        # Action space: 12 discrete actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: 60 floats, all normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(OBS_SIZE, dtype=np.float32),
            high=np.ones(OBS_SIZE,  dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state initialised in reset()
        self.reports       = []   # list of dicts
        self.institutions  = []   # list of dicts
        self.trust         = INITIAL_TRUST
        self.step_count    = 0
        self.resolved_today     = 0
        self.critical_pending   = 0
        self.total_reward       = 0.0
        self.last_action_label  = "—"
        self.event_log          = []   # list of strings for renderer

        # Renderer (lazy init to avoid pygame import on server training)
        self._renderer = None

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Initialise report queue with 5–7 starter reports
        self.reports = []
        n_start = rng.integers(5, 8)
        for _ in range(n_start):
            self.reports.append(self._spawn_report(rng))

        # Pad queue to NUM_REPORTS with empty slots
        while len(self.reports) < NUM_REPORTS:
            self.reports.append(self._empty_report())

        # Initialise institutions
        self.institutions = []
        for i in range(NUM_INSTITUTIONS):
            self.institutions.append({
                "name":            INSTITUTIONS[i],
                "capacity":        rng.uniform(0.5, 1.0),
                "resolution_rate": rng.uniform(0.6, 0.95),
                "workload":        rng.integers(0, 3),
                "response_time":   rng.uniform(1.0, 5.0),
            })

        self.trust            = INITIAL_TRUST
        self.step_count       = 0
        self.resolved_today   = 0
        self.critical_pending = 0
        self.total_reward     = 0.0
        self.last_action_label = "Episode start"
        self.event_log        = ["[ System initialised — dispatch agent online ]"]

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: int):
        self.step_count   += 1
        self.resolved_today = 0
        reward = 0.0

        # ── Decode action ────────────────────────────────────────────────────
        if action in range(0, 4):
            # Dispatch slot-0 report to institution (action)
            reward += self._dispatch(slot=0, inst_idx=action)

        elif action == 4:
            # Escalate the most urgent pending report
            reward += self._escalate()

        elif action == 5:
            # Request more evidence (improves evidence quality of top report)
            reward += self._request_evidence()

        elif action == 6:
            # Defer the lowest-urgency report (small penalty — it's still waiting)
            reward += self._defer()

        elif action == 7:
            # Close top report as duplicate/invalid
            reward += self._close_duplicate()

        elif action in range(8, 12):
            # Dispatch slot-1 report to institution (action - 8)
            reward += self._dispatch(slot=1, inst_idx=action - 8)

        # ── Environment dynamics each step ───────────────────────────────────

        # Age all waiting reports (urgency creep)
        reward += self._age_reports()

        # Random new reports arrive (20% chance per step)
        self._maybe_spawn_report()

        # Institutions process their workload
        self._update_institutions()

        # Trust evolves based on resolution vs neglect
        trust_delta = self._update_trust()
        reward += trust_delta * 0.5   # small trust reward each step

        # Global penalty for every critical report still pending
        self.critical_pending = sum(
            1 for r in self.reports if r["active"] and r["urgency"] >= 4
        )
        reward -= self.critical_pending * 1.5

        # ── Terminal conditions ───────────────────────────────────────────────
        terminated = self.trust < TRUST_COLLAPSE   # system collapse
        truncated  = self.step_count >= MAX_STEPS  # natural end of month

        if terminated:
            reward -= 50.0   # large penalty for letting trust collapse
            self._log(f"⚠ SYSTEM COLLAPSE — trust fell to {self.trust:.1f}%")

        self.total_reward += reward

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Actions ──────────────────────────────────────────────────────────────

    def _dispatch(self, slot: int, inst_idx: int) -> float:
        """Send the report in `slot` to institution `inst_idx`."""
        if slot >= len(self.reports) or not self.reports[slot]["active"]:
            self._log(f"✗ Dispatch slot {slot} — no active report")
            return -3.0   # penalty for wasted action

        report = self.reports[slot]
        inst   = self.institutions[inst_idx]

        # Correct institution match bonus
        correct = (CATEGORY_TO_INST[report["category"]] == inst_idx)
        match_bonus = 5.0 if correct else -2.0

        # Resolution reward: urgency × speed (inverse of wait time)
        speed  = max(0.1, 1.0 - (report["days_waiting"] / 30.0))
        r      = (report["urgency"] * 10.0) * speed + match_bonus

        # Evidence quality multiplier
        r *= (0.7 + 0.3 * report["evidence"])

        # Institution capacity check
        if inst["capacity"] < 0.2:
            r *= 0.4   # overloaded institution — partial resolution
            self._log(f"⚠ {inst['name']} overloaded — partial resolve")

        # Update institution workload
        inst["workload"] = min(5, inst["workload"] + 1)
        inst["capacity"] = max(0.0, inst["capacity"] - 0.1)

        # Resolved — remove from queue
        self.resolved_today += 1
        self.reports[slot] = self._empty_report()

        label = f"→ Dispatched [{CATEGORY_NAMES[report['category']]}] urgency {report['urgency']} to {inst['name']} | +{r:.1f}"
        self.last_action_label = label
        self._log(label)
        return r

    def _escalate(self) -> float:
        """Escalate the most urgent active report."""
        candidates = [(i, r) for i, r in enumerate(self.reports) if r["active"]]
        if not candidates:
            self._log("✗ Escalate — no active reports")
            return -2.0

        idx, report = max(candidates, key=lambda x: x[1]["urgency"])
        old_urgency = report["urgency"]
        report["urgency"] = min(5, report["urgency"] + 1)

        label = f"↑ Escalated [{CATEGORY_NAMES[report['category']]}] urgency {old_urgency}→{report['urgency']}"
        self.last_action_label = label
        self._log(label)
        return 3.0

    def _request_evidence(self) -> float:
        """Request more evidence for the top report — improves routing quality."""
        candidates = [(i, r) for i, r in enumerate(self.reports) if r["active"]]
        if not candidates:
            self._log("✗ Request evidence — no active reports")
            return -2.0

        idx, report = max(candidates, key=lambda x: x[1]["urgency"])
        report["evidence"] = min(1.0, report["evidence"] + 0.3)

        label = f"📎 Evidence requested for [{CATEGORY_NAMES[report['category']]}] — quality now {report['evidence']:.1f}"
        self.last_action_label = label
        self._log(label)
        return 1.5

    def _defer(self) -> float:
        """Defer the lowest-urgency report (move it down the queue)."""
        candidates = [(i, r) for i, r in enumerate(self.reports) if r["active"]]
        if not candidates:
            self._log("✗ Defer — no active reports")
            return -2.0

        idx, report = min(candidates, key=lambda x: x[1]["urgency"])
        report["days_waiting"] += 2   # extra aging penalty

        label = f"⏸ Deferred [{CATEGORY_NAMES[report['category']]}] urgency {report['urgency']}"
        self.last_action_label = label
        self._log(label)
        return -1.0   # small penalty — deferral is sometimes necessary

    def _close_duplicate(self) -> float:
        """Close the top report as a duplicate or invalid."""
        candidates = [(i, r) for i, r in enumerate(self.reports) if r["active"]]
        if not candidates:
            self._log("✗ Close — no active reports")
            return -2.0

        # Only beneficial if evidence is low (likely invalid)
        idx, report = candidates[0]
        r = 2.0 if report["evidence"] < 0.3 else -4.0   # penalty for closing valid reports

        label = f"✗ Closed [{CATEGORY_NAMES[report['category']]}] as duplicate | {'valid close' if r > 0 else 'WRONG — had evidence!'}"
        self.last_action_label = label
        self._log(label)
        self.reports[idx] = self._empty_report()
        return r

    # ── Dynamics ─────────────────────────────────────────────────────────────

    def _age_reports(self) -> float:
        """Age all active reports — urgency creep and neglect penalties."""
        penalty = 0.0
        for report in self.reports:
            if not report["active"]:
                continue
            report["days_waiting"] += 1
            # Critical reports neglected for too long hurt trust
            if report["urgency"] >= 4 and report["days_waiting"] > 7:
                penalty -= 2.0
            # Urgency can creep up if waiting too long
            if report["days_waiting"] > 10 and report["urgency"] < 5:
                if random.random() < 0.1:
                    report["urgency"] += 1
        return penalty

    def _maybe_spawn_report(self):
        """20% chance each step a new citizen report arrives."""
        if random.random() < 0.20:
            empty_slots = [i for i, r in enumerate(self.reports) if not r["active"]]
            if empty_slots:
                slot = random.choice(empty_slots)
                self.reports[slot] = self._spawn_report(self.np_random)
                self._log(f"+ New report: [{CATEGORY_NAMES[self.reports[slot]['category']]}] urgency {self.reports[slot]['urgency']}")

    def _update_institutions(self):
        """Institutions recover capacity over time, workload decreases."""
        for inst in self.institutions:
            inst["capacity"]   = min(1.0, inst["capacity"] + 0.05)
            inst["workload"]   = max(0,   inst["workload"] - 1)
            inst["response_time"] = max(1.0, inst["response_time"] - 0.1)

    def _update_trust(self) -> float:
        """
        Trust evolves based on:
          + resolutions today
          − critical reports unresolved
          − long wait times
        Returns the delta (positive = good for agent).
        """
        delta = 0.0
        delta += self.resolved_today * 2.0
        delta -= self.critical_pending * 1.0
        avg_wait = np.mean([r["days_waiting"] for r in self.reports if r["active"]] or [0])
        delta -= avg_wait * 0.1

        self.trust = np.clip(self.trust + delta, 0.0, 100.0)
        return delta

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _spawn_report(self, rng) -> dict:
        return {
            "active":       True,
            "urgency":      int(rng.integers(1, 6)),         # 1–5
            "category":     int(rng.integers(0, NUM_INSTITUTIONS)),  # 0–3
            "days_waiting": int(rng.integers(0, 5)),
            "evidence":     float(rng.uniform(0.0, 1.0)),
        }

    def _empty_report(self) -> dict:
        return {
            "active":       False,
            "urgency":      0,
            "category":     0,
            "days_waiting": 0,
            "evidence":     0.0,
        }

    def _log(self, msg: str):
        entry = f"Step {self.step_count:03d} | {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > 6:   # keep last 6 for renderer
            self.event_log.pop(0)

    # ── Observation & Info ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        ptr = 0

        # Reports: 10 × 4 features (all normalised 0–1)
        for r in self.reports:
            obs[ptr]     = r["urgency"] / 5.0
            obs[ptr + 1] = r["category"] / (NUM_INSTITUTIONS - 1)
            obs[ptr + 2] = min(r["days_waiting"], 30) / 30.0
            obs[ptr + 3] = r["evidence"]
            ptr += 4

        # Institutions: 4 × 4 features
        for inst in self.institutions:
            obs[ptr]     = inst["capacity"]
            obs[ptr + 1] = inst["resolution_rate"]
            obs[ptr + 2] = inst["workload"] / 5.0
            obs[ptr + 3] = inst["response_time"] / 10.0
            ptr += 4

        # Global
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


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from environment.custom_env import CivicIssueEnv

    env = CivicIssueEnv(render_mode="human")
    obs, info = env.reset(seed=0)

    # Manually create renderer since render() is lazy
    from environment.rendering import CivicRenderer
    renderer = CivicRenderer(width=1100, height=680)

    print("Running random agent visualization — close the window to stop.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        renderer.render(info)

        if term or trunc:
            obs, info = env.reset()

        pygame.time.wait(120)

    renderer.close()
    env.close()