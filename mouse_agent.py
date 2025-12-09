from stable_baselines3 import PPO
from mouse_env import MouseEnv


def train_mouse():
    env = MouseEnv()

    print(" Training PPO Mouse (avoid cat, collect cheese, escape)...")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.02,     # encourage exploration
        verbose=1,
    )

    # You can adjust this number based on time
    model.learn(total_timesteps=400_000)

    model.save("mouse_model.zip")
    print(" Mouse model saved as mouse_model.zip")


if __name__ == "__main__":
    train_mouse()

