import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CivicReportingEnv(gym.Env):
    """
    RL Environment for Rwanda Civic Issue Redress.
    Institutions: 0: NLA, 1: WASAC, 2: REG, 3: MinInfra
    """
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None):
        super(CivicReportingEnv, self).__init__()
        
        self.institutions = ["NLA", "WASAC", "REG", "MinInfra"]
        self.num_inst = len(self.institutions)
        self.max_urgency = 5
        self.max_steps = 100
        
        # Action Space: 0 (Stay/Wait), 1: NLA, 2: WASAC, 3: REG, 4: MinInfra
        self.action_space = spaces.Discrete(self.num_inst + 1)

        # Observation: [Urgency_i, Time_i] for each inst + Agent_Position
        # Size: (4 * 2) + 1 = 9
        low = np.array([0, 0] * self.num_inst + [0], dtype=np.float32)
        high = np.array([self.max_urgency, self.max_steps] * self.num_inst + [self.num_inst - 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.current_step = 0
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial issues at 2 institutions
        obs = np.zeros(self.num_inst * 2 + 1, dtype=np.float32)
        for _ in range(2):
            idx = random.randint(0, self.num_inst - 1)
            obs[idx * 2] = random.randint(2, self.max_urgency)
        
        self.state = obs
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        reward = 0
        agent_pos_idx = self.num_inst * 2
        prev_pos = int(self.state[agent_pos_idx])

        if action > 0:
            target_idx = action - 1
            urgency = self.state[target_idx * 2]
            wait_time = self.state[target_idx * 2 + 1]

            # Travel Penalty: Moving between institutions costs resources
            if target_idx != prev_pos:
                reward -= 2 

            if urgency > 0:
                # Accountability Reward: Priority * Speed
                reward += (urgency * 15) + (wait_time * 0.5)
                self.state[target_idx * 2] = 0 # Issue Resolved
                self.state[target_idx * 2 + 1] = 0
            else:
                reward -= 5 # Penalty for idle dispatch
            
            self.state[agent_pos_idx] = target_idx
        else:
            reward -= 1 # Penalty for total inaction

        # Aging and Spawning
        for i in range(self.num_inst):
            if self.state[i * 2] > 0:
                self.state[i * 2 + 1] += 1
                # Penalty for neglecting high urgency (Accountability Gap)
                if self.state[i * 2] >= 4 and self.state[i * 2 + 1] > 5:
                    reward -= 3

        if random.random() < 0.2: # New issue pops up
            spawn = random.randint(0, self.num_inst - 1)
            if self.state[spawn * 2] == 0:
                self.state[spawn * 2] = random.randint(1, self.max_urgency)

        terminated = False
        truncated = self.current_step >= self.max_steps
        return self.state, reward, terminated, truncated, {}