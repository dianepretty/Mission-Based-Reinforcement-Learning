import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CivicReportingEnv(gym.Env):
    """
    A Custom Environment for Civic Issue Redress in Rwanda.
    The agent acts as a Dispatcher, routing citizen reports to:
    0: NLA (Land)
    1: WASAC (Water)
    2: REG (Electricity)
    3: MinInfra (Infrastructure)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(CivicReportingEnv, self).__init__()
        
        self.render_mode = render_mode

        # Action Space: 4 Discrete Institutions
        # 0: NLA, 1: WASAC, 2: REG, 3: MinInfra
        self.action_space = spaces.Discrete(4)

        # Observation Space: [Issue_Category, Urgency_Level, Wait_Time, Institution_Backlog]
        # Category: 0-3 (Matching the institutions)
        # Urgency: 1-10 (10 being most critical)
        # Wait Time: 0-100 (Steps since reported)
        # Backlog: 0-20 (Number of pending tasks in that institution)
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 0]), 
            high=np.array([3, 10, 100, 20]), 
            dtype=np.float32
        )

        self.state = None
        self.steps_taken = 0
        self.max_steps = 100
        self.backlogs = [0, 0, 0, 0] # Tracks load per institution

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate a random new issue
        category = random.randint(0, 3)
        urgency = random.randint(1, 10)
        wait_time = 0
        avg_backlog = sum(self.backlogs) / 4
        
        self.state = [category, urgency, wait_time, avg_backlog]
        self.steps_taken = 0
        self.backlogs = [0, 0, 0, 0]
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        category, urgency, wait_time, _ = self.state
        
        reward = 0
        terminated = False
        
        # 1. Logic: Correct Routing
        if action == category:
            # Success reward scaled by urgency (high urgency = higher reward for speed)
            reward += (2.0 * urgency)
            self.backlogs[action] = max(0, self.backlogs[action] - 1)
        else:
            # Penalty for misallocation (Wrong institution)
            reward -= 5.0
            self.backlogs[action] += 1 # Adding to wrong backlog increases chaos
            
        # 2. Accountability Penalty: Wait time hurts the citizen experience
        reward -= (wait_time * 0.1)

        # 3. Efficiency Penalty: If backlogs get too high
        if self.backlogs[action] > 10:
            reward -= 2.0

        # Update State for next step:
        # Simulate a new issue arriving
        new_category = random.randint(0, 3)
        new_urgency = random.randint(1, 10)
        new_wait_time = random.randint(0, 5) # New issues start fresh
        new_avg_backlog = sum(self.backlogs) / 4
        
        self.state = [new_category, new_urgency, new_wait_time, new_avg_backlog]

        # Terminal conditions
        if self.steps_taken >= self.max_steps:
            terminated = True
            
        # Truncated is not used here but required by Gymnasium API
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            mapping = {0: "NLA", 1: "WASAC", 2: "REG", 3: "MinInfra"}
            print(f"Step: {self.steps_taken} | Issue: {mapping[int(self.state[0])]} "
                  f"| Urgency: {self.state[1]} | Backlog: {self.state[3]:.1f}")

    def close(self):
        pass