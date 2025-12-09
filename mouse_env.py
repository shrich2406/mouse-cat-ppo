import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mouse_cat_core_env import MouseCatCoreEnv

class MouseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.core = MouseCatCoreEnv()

        self.action_space = spaces.Discrete(4)

        # Matches 8-dim observation
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(8,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self.core.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.core.step(int(action))
        return obs, reward, done, False, info
