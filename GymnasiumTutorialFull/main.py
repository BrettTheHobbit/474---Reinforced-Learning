import time
import gymnasium as gym
import simple_coverage

env = gym.make("complete", render_mode="human")
env.reset()
time.sleep(2)

policy = [0, 1, 4, 1, 2, 3, 2, 3, 3, 3, 3, 0]

for action in policy:
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(1)

env.close()
