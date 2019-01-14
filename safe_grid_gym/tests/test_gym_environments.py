import unittest
import gym
from ai_safety_gridworlds.helpers.factory import _environment_classes
from safe_grid_gym import to_gym_id


class GymEnvironemntTestCase(unittest.TestCase):
    def testMakingGymEnvironments(self):
        """
        Just check that we do not get any errors when making the environments.
        """
        safety_gridworlds = [
            to_gym_id(name) + "-v0" for name in _environment_classes.keys()
        ]

        safety_gridworlds.append("TransitionBoatRace-v0")

        toy_gridworlds = [
            "ToyGridworldUncorrupted-v0",
            "ToyGridworldCorners-v0",
            "ToyGridworldOnTheWay-v0",
        ]

        for gym_env_id in safety_gridworlds + toy_gridworlds:
            gym.make(gym_env_id)
