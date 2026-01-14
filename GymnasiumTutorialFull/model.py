from stable_baselines3 import DQN, A2C
import gymnasium as gym
import time
import simple_coverage
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments for training
vec_env = make_vec_env("complete", n_envs=4)
model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=10000, tb_log_name="dqn_run_1_1")
model.save("gridworld1_1")

# Single environment for testing
env = gym.make("complete", render_mode="human")
model = A2C.load("gridworld1_1")
obs, _ = env.reset()
terminated = False
total_reward = 0

while not terminated:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    total_reward += rewards
    time.sleep(0.5)

print(f"Total reward: {total_reward}")
env.close()
