<img width="897" height="892" alt="image" src="https://github.com/user-attachments/assets/dcf1c757-3668-44b1-845b-bfc8d4a9f34b" />

Demo
https://github.com/user-attachments/assets/2350b192-098b-45c5-a6ff-a2d6d4ed5c9c
# 🐭🐱 Mouse–Cat PPO Reinforcement Learning Environment

This project implements a custom reinforcement learning environment where a **mouse agent** must navigate a grid, collect cheese, avoid traps, and reach an exit — all while being chased by a deterministic **cat**. The mouse is trained using **Proximal Policy Optimization (PPO)**, and the environment includes a **Pygame UI** for visualization.

---

## 🚀 Project Overview

The Mouse–Cat environment is designed to explore RL behavior in a partially adversarial grid world:

- The **mouse agent** is trained with PPO to reach the exit.
- The **cat** is not an RL agent — it follows a deterministic chase movement.
- The grid includes:
  - Randomized trap tiles  
  - Randomized cheese tiles  
  - Fixed start and exit  
- The mouse moves **2 steps per turn**, and the cat moves **1 chasing step**.

This environment supports RL experimentation, curriculum design, and gameplay visualization.

---

## 📁 Repository Structure
mouse-cat-ppo/
│
├── assets/ # Screenshots or UI assets
├── mouse_agent.py # PPO training script for the mouse
├── mouse_env.py # Gym wrapper for RL agent
├── mouse_cat_core_env.py # Core environment logic
├── pygame_ui.py # Pygame visualization interface
├── .gitignore
└── README.md
## 🧠 Environment Design

### Grid Rules
- Default grid size: **12 × 12**
- Mouse start: `(0, grid_size // 2)`
- Exit: `(grid_size − 1, grid_size // 2)`
- Random traps and cheese spawn each episode.

### Mouse Movement
- Moves **two single-steps** each turn.
- Dies only if its final landing tile is a trap.
- Reward-driven pathfinding.

### Cat Movement
- Moves **one step per turn**
- Always moves toward mouse's position (greedy chasing)
- Touching cat ends episode with penalty.

### Episode Termination
- Mouse reaches exit  
- Mouse steps on trap  
- Cat catches mouse  
- Max-step limit reached  

---

## 🎯 Action & Observation Space

### Action Space
Mouse can choose from 4 discrete actions:
0 → Up
1 → Down
2 → Left
3 → Right
### Observation Space
The observation vector includes:
- Mouse position  
- Cat position  
- Trap locations  
- Cheese locations  
- Exit tile location  
- Remaining steps  
- Flattened into a numeric state for PPO  

---

## 💰 Reward Structure

| Event            | Reward |
|------------------|--------|
| Reach exit       | +100   |
| Collect cheese   | +10    |
| Step on trap     | -10    |
| Caught by cat    | -100   |
| Each step taken  | -1     |

These values can be adjusted in `mouse_cat_core_env.py` to tune agent behavior.

---

## 🏋️ Training the PPO Agent

To train the mouse agent with PPO, run:

```bash
python mouse_agent.py
