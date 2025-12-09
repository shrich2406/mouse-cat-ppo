from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from mouse_env import MouseEnv


def train_mouse():
    print("🚀 Training PPO Mouse (avoid cat, collect cheese, escape)...")

    env = Monitor(MouseEnv())
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        n_epochs=10,
    )

    model.learn(total_timesteps=400_000)

    print("🎉 Saving model...")
    model.save("mouse_model")

    print("✔ Training complete")


if __name__ == "__main__":
    train_mouse()
