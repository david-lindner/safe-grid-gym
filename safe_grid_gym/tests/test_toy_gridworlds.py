import unittest
import gym
import numpy as np
import matplotlib


class ToyGridworldsTestCase(unittest.TestCase):
    def testRenderReturnsNotImplemented(self):
        """ If called with an undefined mode argument the render function should
        return an error. """

        toy_gridworlds = [
            "ToyGridworldUncorrupted-v0",
            "ToyGridworldCorners-v0",
            "ToyGridworldOnTheWay-v0",
        ]

        for gym_env_id in toy_gridworlds:
            env = gym.make(gym_env_id)

            # should give no error
            self.assertIsInstance(env.render(mode="ansi"), str)
            self.assertIsInstance(env.render(mode="rgb_array"), np.ndarray)

            matplotlib.use("Agg")
            env.render(mode="human")

            # "text" is not a valid mode
            with self.assertRaises(NotImplementedError):
                env.render(mode="text")
