# Rwanda Civic Dispatch — Reinforcement Learning

A reinforcement learning system that trains an agent to optimally dispatch,
escalate, and manage citizen reports across Rwanda's government institutions.

## Mission

Tackle citizen disengagement and lack of government accountability by training
an RL agent to connect citizens with responsible government institutions,
automatically prioritize issues by urgency, and track resolution with
evidence-based accountability.

## Project Structure

```
CivicRL/
├── environment/
│   ├── custom_env.py      # Custom Gymnasium environment
│   └── rendering.py       # Pygame dashboard visualization
├── training/
│   ├── dqn_training.py    # DQN — 10 hyperparameter experiments
│   ├── pg_training.py     # PPO + A2C — 10 experiments each
│   └── reinforce_training.py  # REINFORCE — 10 experiments
├── models/
│   ├── dqn/               # Saved DQN models
│   ├── ppo/               # Saved PPO models
│   ├── a2c/               # Saved A2C models
│   ├── reinforce/         # Saved REINFORCE models
│   └── best_model/        # Best performing model
├── logs/                  # Training logs and reward CSVs
├── main.py                # Run best trained agent
├── demo.py                # Random agent visualization
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. View random agent demo (no training needed):**
```bash
python demo.py
```

**2. Train all algorithms:**
```bash
python training/dqn_training.py
python training/pg_training.py
python training/reinforce_training.py
```

**3. Run best trained agent:**
```bash
python main.py
python main.py --algo ppo   # specify algorithm
```

## Environment

| Property | Value |
|---|---|
| Observation space | Box(60,) — normalised floats |
| Action space | Discrete(12) |
| Max steps | 200 (1 simulated month) |
| Terminal condition | Trust < 20% (system collapse) |

### Actions
| ID | Action |
|---|---|
| 0–3 | Dispatch slot-0 report to NLA / WASAC / REG / MININFRA |
| 4 | Escalate most urgent pending report |
| 5 | Request more evidence from citizen |
| 6 | Defer lowest-urgency report |
| 7 | Close report as duplicate/invalid |
| 8–11 | Dispatch slot-1 report to NLA / WASAC / REG / MININFRA |

### Reward Structure
- `+ urgency × speed` — resolving reports quickly
- `+ trust_delta × 0.5` — citizen trust growing
- `− neglect penalty` — critical reports left unresolved
- `− 50` — trust collapse (episode ends early)

## Algorithms Compared

| Algorithm | Type | Key Strength |
|---|---|---|
| DQN | Value-based | Experience replay, stable Q-learning |
| PPO | Policy gradient | Clipped updates, sample efficient |
| A2C | Actor-critic | Fast updates, good for dense rewards |
| REINFORCE | Policy gradient | Pure Monte Carlo, simple and clean |
