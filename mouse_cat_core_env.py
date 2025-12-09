import numpy as np
import random


class MouseCatCoreEnv:
    """
    Final version:
    - Mouse moves 2 single-steps
    - Cat moves 1 chasing step
    - Start fixed, exit fixed
    - Cheeses & traps randomized each game
    - Mouse dies only if final landing tile is trap
    """

    def __init__(self, grid_size=12, n_traps=6, n_cheese=3, max_steps=350):
        self.grid_size = grid_size
        self.n_traps = n_traps
        self.n_cheese = n_cheese
        self.max_steps = max_steps
        self.reset()

    def _sample_pos(self, forbidden):
        while True:
            p = (random.randint(0, self.grid_size - 1),
                 random.randint(0, self.grid_size - 1))
            if p not in forbidden:
                return p

    def reset(self):
        forbidden = set()

        # Fixed start & exit
        self.start_pos = (0, self.grid_size // 2)
        self.exit = (self.grid_size - 1, self.grid_size // 2)
        forbidden.add(self.start_pos)
        forbidden.add(self.exit)

        # Cat fixed near mouse
        self.cat_start = (1, self.grid_size // 2)
        forbidden.add(self.cat_start)

        # Random cheeses
        self.cheeses = []
        self.cheese_alive = []
        for _ in range(self.n_cheese):
            c = self._sample_pos(forbidden)
            self.cheeses.append(c)
            self.cheese_alive.append(True)
            forbidden.add(c)

        # Random traps
        self.traps = set()
        for _ in range(self.n_traps):
            t = self._sample_pos(forbidden)
            self.traps.add(t)

        # Dynamic states
        self.mouse = self.start_pos
        self.cat = self.cat_start
        self.steps = 0

        return self.get_obs()

    def get_obs(self):
        flat_cheese = []
        for c in self.cheeses:
            flat_cheese.extend([c[0], c[1]])

        return np.array([
            self.mouse[0], self.mouse[1],
            self.cat[0], self.cat[1],
            self.exit[0], self.exit[1],
            *flat_cheese,
            int(any(self.cheese_alive))
        ], dtype=np.float32)

    def _move_single_step(self, pos, action):
        x, y = pos
        if action == 0: y -= 1   # up
        elif action == 1: y += 1 # down
        elif action == 2: x -= 1 # left
        elif action == 3: x += 1 # right

        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        return (x, y)

    def step(self, action_mouse):
        reward = -0.15
        self.steps += 1

        # -------- Mouse moves 2 substeps --------
        for _ in range(2):
            new_mouse = self._move_single_step(self.mouse, action_mouse)

            for i, pos in enumerate(self.cheeses):
                if self.cheese_alive[i] and new_mouse == pos:
                    self.cheese_alive[i] = False
                    reward += 4

            self.mouse = new_mouse

        # -------- Cat moves 1 step --------
        cx, cy = self.cat
        mx, my = self.mouse

        if cx < mx: cx += 1
        elif cx > mx: cx -= 1
        if cy < my: cy += 1
        elif cy > my: cy -= 1

        self.cat = (cx, cy)

        # -------- Terminal checks --------
        if self.mouse == self.cat:
            return self.get_obs(), -12, True, {"result": "caught"}

        if self.mouse in self.traps:
            return self.get_obs(), -12, True, {"result": "trap"}

        if self.mouse == self.exit:
            bonus = 6 if not all(self.cheese_alive) else 0
            return self.get_obs(), 10 + bonus, True, {"result": "escaped"}

        if self.steps >= self.max_steps:
            return self.get_obs(), -3, True, {"result": "timeout"}

        return self.get_obs(), reward, False, {}

