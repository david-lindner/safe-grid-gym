import unittest
import gym
import numpy as np
import matplotlib

TOY_GRIDWORLDS = [
    "ToyGridworldUncorrupted-v0",
    "ToyGridworldCorners-v0",
    "ToyGridworldOnTheWay-v0",
]


class ToyGridworldsTestCase(unittest.TestCase):
    def testRenderReturnsNotImplemented(self):
        """ If called with an undefined mode argument the render function should
        return an error. """

        for gym_env_id in TOY_GRIDWORLDS:
            env = gym.make(gym_env_id)

            # should give no error
            self.assertIsInstance(env.render(mode="ansi"), str)
            self.assertIsInstance(env.render(mode="rgb_array"), np.ndarray)

            matplotlib.use("Agg")
            env.render(mode="human")

            # "text" is not a valid mode
            with self.assertRaises(NotImplementedError):
                env.render(mode="text")

    def testWithRandomMoves(self):
        """ Just makes sure that nothing crashes if we make random moves. """
        for gym_env_id in TOY_GRIDWORLDS:
            env = gym.make(gym_env_id)

            env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
