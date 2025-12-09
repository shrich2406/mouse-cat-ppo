import gymnasium as gym
from gymnasium import spaces
import numpy as np

from mouse_cat_core_env import MouseCatCoreEnv


class CatEnv(gym.Env):
    """
    Gym wrapper where:
    - Cat is the RL agent
    - Mouse follows a simple 'go to cheese then exit' heuristic
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size=10, max_steps=120):
        super().__init__()

        self.core = MouseCatCoreEnv(grid_size=grid_size, n_traps=5, max_steps=max_steps)

        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(9,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.core.reset()
        return obs.astype(np.float32), {}

    def _mouse_greedy_action(self):
        mx, my = self.core.mouse
        if not self.core.has_cheese:
            tx, ty = self.core.cheese
        else:
            tx, ty = self.core.exit

        dx = tx - mx
        dy = ty - my

        if abs(dx) > abs(dy):
            if dx > 0:
                return 3  # right
            else:
                return 2  # left
        else:
            if dy > 0:
                return 1  # down
            else:
                return 0  # up

    def step(self, action):
        mouse_action = self._mouse_greedy_action()
        obs, r_mouse, r_cat, done, info = self.core.step(mouse_action, int(action))
        return obs.astype(np.float32), float(r_cat), bool(done), False, info
