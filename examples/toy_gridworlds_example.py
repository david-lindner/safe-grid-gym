import gym
import safe_grid_gym

# env = gym.make("ToyGridworldUncorrupted-v0")
env = gym.make("ToyGridworldCorners-v0")
# env = gym.make("ToyGridworldOnTheWay-v0")

env.reset()

for i in range(100):
    env.render()
    action = env.action_space.sample()
    env.step(action)
