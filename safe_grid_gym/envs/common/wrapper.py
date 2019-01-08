import gym

from safe_grid_gym.envs.common.interface import INFO_HIDDEN_REWARD

class Cheat(gym.Wrapper):

    def __init__(self, env):
        self.cheat = True
        super().__init__(env)

    def step(self, action):
        if not self.cheat:
            return self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        if INFO_HIDDEN_REWARD in info:
            return obs, info[INFO_HIDDEN_REWARD], done, info
        else:
            error.Error("This environment does not support cheating.")
            self.cheat = False
            return obs, reward, done, info

    def reset(self):
        return selv.env.reset()
