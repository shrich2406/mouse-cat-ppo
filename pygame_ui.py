import pygame
import sys
from mouse_cat_core_env import MouseCatCoreEnv
from stable_baselines3 import PPO
import numpy as np

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------
CELL_SIZE = 60
GRID_COLOR = (210, 210, 210)
BG_COLOR = (245, 245, 245)

USE_PPO = True  # set False for random movement

pygame.init()
screen = pygame.display.set_mode((900, 900))
pygame.display.set_caption("Mouse vs Cat - PPO Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 26, bold=True)

# -----------------------------------------------------------
# Load sprites
# -----------------------------------------------------------
def load_sprite(path):
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))

mouse_img = load_sprite("assets/mouse.png")
cat_img = load_sprite("assets/cat.png")
cheese_img = load_sprite("assets/cheese.png")
trap_img = load_sprite("assets/trap.png")
start_img = load_sprite("assets/start.png")
exit_img = load_sprite("assets/exit.png")

# -----------------------------------------------------------
# Load PPO model
# -----------------------------------------------------------
model = None
if USE_PPO:
    try:
        print("📦 Loading PPO model mouse_model.zip ...")
        model = PPO.load("mouse_model.zip")
        print("✅ PPO model loaded! (stochastic policy)")
    except:
        print("❌ PPO model missing, switching to random mode...")
        USE_PPO = False


# -----------------------------------------------------------
# Draw grid elements
# -----------------------------------------------------------
def draw(env):
    screen.fill(BG_COLOR)

    # grid
    for x in range(env.grid_size):
        pygame.draw.line(screen, GRID_COLOR, (x * CELL_SIZE, 0),
                         (x * CELL_SIZE, env.grid_size * CELL_SIZE))
        pygame.draw.line(screen, GRID_COLOR, (0, x * CELL_SIZE),
                         (env.grid_size * CELL_SIZE, x * CELL_SIZE))

    # start + exit
    screen.blit(start_img, (env.start_pos[0] * CELL_SIZE, env.start_pos[1] * CELL_SIZE))
    screen.blit(exit_img, (env.exit[0] * CELL_SIZE, env.exit[1] * CELL_SIZE))

    # cheeses
    for i, pos in enumerate(env.cheeses):
        if env.cheese_alive[i]:
            screen.blit(cheese_img, (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE))

    # traps
    for (tx, ty) in env.traps:
        screen.blit(trap_img, (tx * CELL_SIZE, ty * CELL_SIZE))

    # mouse and cat
    screen.blit(mouse_img, (env.mouse[0] * CELL_SIZE, env.mouse[1] * CELL_SIZE))
    screen.blit(cat_img, (env.cat[0] * CELL_SIZE, env.cat[1] * CELL_SIZE))


# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
def run():
    env = MouseCatCoreEnv()
    obs = env.reset()

    running = True

    while running:
        clock.tick(7)  # FPS

        # -----------------------------------
        # PPO or random action
        # -----------------------------------
        if USE_PPO and model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = np.random.randint(0, 4)

        # -----------------------------------
        # STEP THE ENVIRONMENT (FIXED)
        # -----------------------------------
        obs, reward, done, info = env.step(int(action))

        draw(env)

        # text
        reward_text = font.render(f"Reward: {reward}", True, (10, 10, 10))
        screen.blit(reward_text, (10, 750))

        pygame.display.update()

        # End of game
        if done:
            pygame.time.wait(1200)
            obs = env.reset()

        # Check quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    sys.exit()


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    run()
