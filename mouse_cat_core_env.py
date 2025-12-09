import numpy as np
import random

class MouseCatCoreEnv:
    """
    Environment with:
    - Mouse moves 2-step (one-by-one pass check)
    - Cat moves 1-step chase
    - Random cheeses (1–4)
    - Random traps
    - Terminal when mouse passes or lands on trap
    - PPO requires: obs, reward, done, info
    """

    def __init__(self, grid_size=12, max_steps=400):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    # ----------------------------
    # Reset new randomized env
    # ----------------------------
    def reset(self):
        self.steps = 0

        # Mouse starts middle-left
        self.start_pos = (0, self.grid_size // 2)
        self.mouse = self.start_pos

        # Cat starts next to mouse
        self.cat = (1, self.grid_size // 2)

        # Random traps
        self.traps = set()
        for _ in range(random.randint(3, 6)):
            self.traps.add(self._rand_free())

        # Random cheeses
        self.cheeses = []
        self.cheese_alive = []
        for _ in range(random.randint(1, 4)):
            pos = self._rand_free()
            self.cheeses.append(pos)
            self.cheese_alive.append(True)

        # Exit bottom-right
        self.exit = (self.grid_size - 1, self.grid_size - 1)

        return self.get_obs()

    # ----------------------------
    def _rand_free(self):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in [self.mouse, self.cat] and (x, y) not in self.traps:
                return (x, y)

    # ----------------------------
    def get_obs(self):
        """
        Mouse PPO sees:
        mouse_x, mouse_y,
        cat_x, cat_y,
        nearest_cheese_x, nearest_cheese_y,
        exit_x, exit_y
        """
        # nearest cheese
        alive_positions = [c for i, c in enumerate(self.cheeses) if self.cheese_alive[i]]
        if alive_positions:
            nearest = min(alive_positions, key=lambda c: abs(c[0]-self.mouse[0]) + abs(c[1]-self.mouse[1]))
        else:
            nearest = self.exit  # go to exit if cheese all eaten

        return np.array([
            self.mouse[0], self.mouse[1],
            self.cat[0], self.cat[1],
            nearest[0], nearest[1],
            self.exit[0], self.exit[1]
        ], dtype=np.float32)

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, action):
        self.steps += 1

        # Mouse 2-step but one-by-one
        move = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0)
        }

        reward = -0.05
        done = False
        info = {}

        # two mini-steps
        for _ in range(2):
            dx, dy = move[action]
            mx, my = self.mouse

            nx = int(np.clip(mx + dx, 0, self.grid_size - 1))
            ny = int(np.clip(my + dy, 0, self.grid_size - 1))
            self.mouse = (nx, ny)

            # Check pass-through trap
            if self.mouse in self.traps:
                return self.get_obs(), -10, True, {"result": "trap"}

            # Cheese eaten
            for i, pos in enumerate(self.cheeses):
                if self.cheese_alive[i] and self.mouse == pos:
                    self.cheese_alive[i] = False
                    reward += 5

            # Exit condition
            if self.mouse == self.exit:
                return self.get_obs(), 20, True, {"result": "escaped"}

        # Cat moves 1-step
        cx, cy = self.cat
        mx, my = self.mouse

        if cx < mx: cx += 1
        elif cx > mx: cx -= 1
        if cy < my: cy += 1
        elif cy > my: cy -= 1

        self.cat = (cx, cy)

        # Cat catches mouse
        if self.cat == self.mouse:
            return self.get_obs(), -10, True, {"result": "caught"}

        # Timeout
        if self.steps >= self.max_steps:
            return self.get_obs(), -3, True, {"result": "timeout"}

        return self.get_obs(), reward, False, info
