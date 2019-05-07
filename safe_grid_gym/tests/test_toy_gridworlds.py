import unittest
import gym
import numpy as np
import matplotlib
import random

from safe_grid_gym.envs.common.base_gridworld import UP, DOWN, LEFT, RIGHT
from safe_grid_gym.envs.common.interface import INFO_HIDDEN_REWARD

TOY_GRIDWORLDS = [
    "ToyGridworldUncorrupted-v0",
    "ToyGridworldCorners-v0",
    "ToyGridworldOnTheWay-v0",
]


class ToyGridworldsTestCase(unittest.TestCase):
    def _check_rgb(self, rgb_list):
        first_shape = rgb_list[0].shape
        for rgb in rgb_list:
            self.assertIsInstance(rgb, np.ndarray)
            self.assertEqual(len(rgb.shape), 3)
            self.assertEqual(rgb.shape[0], 3)
            self.assertEqual(rgb.shape, first_shape)
            # should only be grayscale
            self.assertTrue(np.all(rgb[0] - rgb[1] == 0))
            self.assertTrue(np.all(rgb[1] - rgb[2] == 0))

    def _check_step(self, res, expected_done, expected_reward, expected_hidden_reward):
        obs, reward, done, info = res
        self.assertEqual(done, expected_done)
        self.assertEqual(reward, expected_reward)
        self.assertEqual(info[INFO_HIDDEN_REWARD], expected_hidden_reward)

    def testRenderReturnsNotImplemented(self):
        """ If called with an undefined mode argument the render function should
        return an error. """

        for gym_env_id in TOY_GRIDWORLDS:
            env = gym.make(gym_env_id)

            # should give no error
            self.assertIsInstance(env.render(mode="ansi"), str)
            rgb = env.render(mode="rgb_array")
            self._check_rgb([rgb])

            matplotlib.use("Agg")
            env.render(mode="human")

            # "text" is not a valid mode
            with self.assertRaises(NotImplementedError):
                env.render(mode="text")

    def testWithRandomMoves(self):
        """ Just makes sure that nothing crashes if we make random moves. """
        for gym_env_id in TOY_GRIDWORLDS:

            N = 3
            actions = []

            for i in range(N):
                random.seed(42)
                np.random.seed(42)
                env = gym.make(gym_env_id)
                env.seed(42)
                env.action_space.seed(42)
                env.observation_space.seed(42)
                obs = env.reset()
                actions.append([])
                done = False

                # test onservation shapes
                self.assertEqual(len(env.observation_space.shape), 3)
                self.assertEqual(len(obs.shape), 3)
                self.assertTrue(env.observation_space.contains(obs))

                while not done:
                    action = env.action_space.sample()
                    actions[i].append(action)
                    obs, reward, done, info = env.step(action)
                    self.assertEqual(len(obs.shape), 3)
                    self.assertTrue(env.observation_space.contains(obs))

                # sampled actions should be the same because each run has the same seed
                self.assertEqual(actions[i], actions[0])

    def testCornersCorrupt(self):
        """ Check reward along some basic trajectories. """
        env = gym.make("ToyGridworldCorners-v0")
        env.reset()

        demonstrations = []

        steps = [UP, UP, UP, UP, LEFT, LEFT, LEFT, LEFT]
        results = [
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 11, 6),
            (False, 7, 7),
            (False, 8, 8),
            (False, 9, 9),
            (True, 10, 10),
        ]

        demonstrations.append((steps, results))

        steps = [UP, LEFT, UP, LEFT, UP, LEFT, UP, LEFT, UP, LEFT]
        results = [
            (False, 6, 6),
            (False, 7, 7),
            (False, 7, 7),
            (False, 8, 8),
            (False, 8, 8),
            (False, 9, 9),
            (False, 9, 9),
            (True, 10, 10),
        ]

        demonstrations.append((steps, results))

        steps = [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT]
        results = [
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (True, 6, 6),
        ]

        demonstrations.append((steps, results))
        rgb_list = []

        for steps, exp_res in demonstrations:
            env.reset()
            for step, exp_res in zip(steps, exp_res):
                res = env.step(step)
                self._check_step(res, *exp_res)
                rgb = env.render(mode="rgb_array")
                rgb_list.append(rgb)

        self._check_rgb(rgb_list)

    def testWayCorrupt(self):
        """ Check reward along some basic trajectories. """
        env = gym.make("ToyGridworldOnTheWay-v0")
        env.reset()

        demonstrations = []

        steps = [UP, UP, UP, UP, LEFT, LEFT, LEFT, LEFT]
        results = [
            (False, 6, 6),
            (False, 6, 6),
            (False, 6, 6),
            (False, 11, 6),
            (False, 7, 7),
            (False, 8, 8),
            (False, 9, 9),
            (True, 10, 10),
        ]

        demonstrations.append((steps, results))

        steps = [UP, LEFT, UP, LEFT, UP, LEFT, UP, LEFT, UP, LEFT]
        results = [
            (False, 6, 6),
            (False, 7, 7),
            (False, 7, 7),
            (False, 8, 8),
            (False, 11, 8),
            (False, 9, 9),
            (False, 9, 9),
            (True, 10, 10),
        ]

        demonstrations.append((steps, results))

        rgb_list = []

        for steps, exp_res in demonstrations:
            env.reset()
            for step, exp_res in zip(steps, exp_res):
                res = env.step(step)
                self._check_step(res, *exp_res)
                rgb = env.render(mode="rgb_array")
                rgb_list.append(rgb)

        self._check_rgb(rgb_list)

    def testObservationSpaceConsistent(self):
        """ Make sure that sampled observations are contained in the observation space. """
        for gym_env_id in TOY_GRIDWORLDS:
            random.seed(42)
            np.random.seed(42)
            env = gym.make(gym_env_id)
            env.seed(42)
            env.observation_space.seed(42)
            N = 20
            for i in range(N):
                obs = env.observation_space.sample()
                self.assertTrue(env.observation_space.contains(obs))
