import gym
import safe_grid_gym

# env = gym.make("ToyGridworldUncorrupted-v0")
env = gym.make("ToyGridworldCorners-v0")
# env = gym.make("ToyGridworldOnTheWay-v0")

env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
