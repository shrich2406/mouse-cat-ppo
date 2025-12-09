from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cat_env import CatEnv

def make_env():
    return CatEnv(grid_size=7, max_steps=120)

def train_cat():
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4096,
        batch_size=1024,
        gamma=0.98,
        gae_lambda=0.92,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        clip_range=0.15,
        tensorboard_log="./cat_tb",
    )

    model.learn(total_timesteps=1_000_000)
    model.save("cat_model.zip")
    print("Cat model saved!")

if __name__ == "__main__":
    train_cat()

