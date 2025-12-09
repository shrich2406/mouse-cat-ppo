import pygame
import sys
from mouse_cat_core_env import MouseCatCoreEnv
from stable_baselines3 import PPO

CELL = 60
pygame.init()
screen = pygame.display.set_mode((900, 900))
font = pygame.font.SysFont("Arial", 26)

mouse_img = pygame.transform.scale(pygame.image.load("assets/mouse.png"), (CELL, CELL))
cat_img = pygame.transform.scale(pygame.image.load("assets/cat.png"), (CELL, CELL))
trap_img = pygame.transform.scale(pygame.image.load("assets/trap.png"), (CELL, CELL))
cheese_img = pygame.transform.scale(pygame.image.load("assets/cheese.png"), (CELL, CELL))
start_img = pygame.transform.scale(pygame.image.load("assets/start.png"), (CELL, CELL))
exit_img = pygame.transform.scale(pygame.image.load("assets/exit.png"), (CELL, CELL))

try:
    print("📦 Loading PPO model mouse_model.zip ...")
    model = PPO.load("mouse_model")
    print("✅ PPO model loaded! (stochastic policy)")
except:
    print("❌ PPO model missing! Exiting.")
    sys.exit()


def draw(env):
    screen.fill((240, 240, 240))

    for x in range(env.grid_size):
        pygame.draw.line(screen, (180, 180, 180), (x * CELL, 0), (x * CELL, env.grid_size * CELL))

    for y in range(env.grid_size):
        pygame.draw.line(screen, (180, 180, 180), (0, y * CELL), (env.grid_size * CELL, y * CELL))

    # Start & exit
    screen.blit(start_img, (env.start_pos[0] * CELL, env.start_pos[1] * CELL))
    screen.blit(exit_img, (env.exit[0] * CELL, env.exit[1] * CELL))

    # Cheeses
    for i, pos in enumerate(env.cheeses):
        if env.cheese_alive[i]:
            screen.blit(cheese_img, (pos[0] * CELL, pos[1] * CELL))

    # Traps
    for t in env.traps:
        screen.blit(trap_img, (t[0] * CELL, t[1] * CELL))

    # Mouse & cat
    screen.blit(mouse_img, (env.mouse[0] * CELL, env.mouse[1] * CELL))
    screen.blit(cat_img, (env.cat[0] * CELL, env.cat[1] * CELL))


def run():
    env = MouseCatCoreEnv()
    obs = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(int(action))

        draw(env)

        reward_msg = font.render(f"Reward: {reward}", True, (0, 0, 0))
        screen.blit(reward_msg, (10, 850))

        pygame.display.update()
        pygame.time.delay(120)

        if done:
            pygame.time.delay(700)
            obs = env.reset()


if __name__ == "__main__":
    run()
