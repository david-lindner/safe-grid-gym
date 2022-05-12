"""
The GridworldEnv implements the gym interface for the ai_safety_gridworlds.

GridworldEnv is based on an implementation by n0p2.
The original repo can be found at https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import importlib
import random
import gym
import copy
import numpy as np

from gym import error
from gym.utils import seeding
from ai_safety_gridworlds.helpers import factory
from safe_grid_gym.viewer import AgentViewer
from safe_grid_gym.envs.common.interface import (
    INFO_HIDDEN_REWARD,
    INFO_OBSERVED_REWARD,
    INFO_DISCOUNT,
)


class GridworldEnv(gym.Env):
    """ An OpenAI Gym environment wrapping the AI safety gridworlds created by DeepMind.

    Parameters:
    env_name (str): defines the safety gridworld to load. can take all values
                    defined in ai_safety_gridworlds.helpers.factory._environment_classes:
                        - 'boat_race'
                        - 'conveyor_belt'
                        - 'distributional_shift'
                        - 'friend_foe'
                        - 'island_navigation'
                        - 'rocks_diamonds'
                        - 'safe_interruptibility'
                        - 'side_effects_sokoban'
                        - 'tomato_watering'
                        - 'tomato_crmdp'
                        - 'absent_supervisor'
                        - 'whisky_gold'
    use_transitions (bool): If set to true the state will be the concatenation
                            of the board at time t-1 and at time t
    render_animation_delay (float): is passed through to the AgentViewer
                                    and defines the speed of the animation in
                                    render mode "human"
    """

    metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, env_name, use_transitions=False, render_animation_delay=0.1, *args, **kwargs):
        self._env_name = env_name
        self._render_animation_delay = render_animation_delay
        self._viewer = None
        self._env = factory.get_environment_obj(env_name, *args, **kwargs)
        self._rbg = None
        self._last_hidden_reward = 0
        self._use_transitions = use_transitions
        self._last_board = None
        self.action_space = GridworldsActionSpace(self._env)
        self.observation_space = GridworldsObservationSpace(self._env, use_transitions)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def step(self, action):
        """ Perform an action in the gridworld environment.

        Returns:
            - the board as a numpy array
            - the observed reward
            - if the episode ended
            - an info dict containing:
                - the observed reward with key INFO_OBSERVED_REWARD
                - the hidden reward with key INFO_HIDDEN_REWARD
                - the discount factor of the last step with key INFO_DISCOUNT
                - any additional information in the pycolab observation object,
                  excluding the RGB array. This includes in particular
                  the "extra_observations"
        """
        timestep = self._env.step(action)
        obs = timestep.observation
        self._rgb = obs["RGB"]

        reward = 0.0 if timestep.reward is None else timestep.reward
        done = timestep.step_type.last()

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            hidden_reward = cumulative_hidden_reward - self._last_hidden_reward
            self._last_hidden_reward = cumulative_hidden_reward
        else:
            hidden_reward = None

        info = {
            INFO_HIDDEN_REWARD: hidden_reward,
            INFO_OBSERVED_REWARD: reward,
            INFO_DISCOUNT: timestep.discount,
        }

        for k, v in obs.items():
            if k not in ("board", "RGB"):
                info[k] = v

        board = copy.deepcopy(obs["board"])

        if self._use_transitions:
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        return (state, reward, done, info)

    def reset(self):
        timestep = self._env.reset()
        self._rgb = timestep.observation["RGB"]
        if self._viewer is not None:
            self._viewer.reset_time()

        board = copy.deepcopy(timestep.observation["board"])

        if self._use_transitions:
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        """ Implements the gym render modes "rgb_array", "ansi" and "human".

        - "rgb_array" just passes through the RGB array provided by pycolab in each state
        - "ansi" gets an ASCII art from pycolab and returns is as a string
        - "human" uses the ai-safety-gridworlds-viewer to show an animation of the
          gridworld in a terminal
        """
        if mode == "rgb_array":
            if self._rgb is None:
                error.Error("environment has to be reset before rendering")
            else:
                return self._rgb
        elif mode == "ansi":
            if self._env._current_game is None:
                error.Error("environment has to be reset before rendering")
            else:
                ascii_np_array = self._env._current_game._board.board
                ansi_string = "\n".join(
                    [
                        " ".join([chr(i) for i in ascii_np_array[j]])
                        for j in range(ascii_np_array.shape[0])
                    ]
                )
                return ansi_string
        elif mode == "human":
            if self._viewer is None:
                self._viewer = init_viewer(self._env_name, self._render_animation_delay)
                self._viewer.display(self._env)
            else:
                self._viewer.display(self._env)
        else:
            super(GridworldEnv, self).render(mode=mode)  # just raise an exception


class GridworldsActionSpace(gym.Space):
    def __init__(self, env):
        action_spec = env.action_spec()
        assert action_spec.name == "discrete"
        assert action_spec.dtype == "int32"
        assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
        self.min_action = action_spec.minimum
        self.max_action = action_spec.maximum
        self.n = (self.max_action - self.min_action) + 1
        super(GridworldsActionSpace, self).__init__(
            shape=action_spec.shape, dtype=action_spec.dtype
        )

    def sample(self):
        return random.randint(self.min_action, self.max_action)

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.Space):
    def __init__(self, env, use_transitions):
        self.observation_spec_dict = env.observation_spec()
        self.use_transitions = use_transitions
        if self.use_transitions:
            shape = (2, *self.observation_spec_dict["board"].shape)
        else:
            shape = (1, *self.observation_spec_dict["board"].shape)
        dtype = self.observation_spec_dict["board"].dtype
        super(GridworldsObservationSpace, self).__init__(shape=shape, dtype=dtype)

    def sample(self):
        """
        Use pycolab to generate an example observation. Note that this is not a
        random sample, but might return the same observation for every call.
        """
        if self.use_transitions:
            raise NotImplementedError(
                "Sampling from transition-based envs not yet supported."
            )
        observation = {}
        for key, spec in self.observation_spec_dict.items():
            if spec == {}:
                observation[key] = {}
            else:
                observation[key] = spec.generate_value()
        return observation["board"][np.newaxis, :]

    def contains(self, x):
        if "board" in self.observation_spec_dict.keys():
            try:
                self.observation_spec_dict["board"].validate(x[0, ...])
                if self.use_transitions:
                    self.observation_spec_dict["board"].validate(x[1, ...])
                return True
            except ValueError:
                return False
        else:
            return False


def init_viewer(env_name, pause):
    (color_bg, color_fg) = get_color_map(env_name)
    av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)
    return av


def get_color_map(env_name):
    module_prefix = "ai_safety_gridworlds.environments."
    env_module_name = module_prefix + env_name
    env_module = importlib.import_module(env_module_name)
    color_bg = env_module.GAME_BG_COLOURS
    color_fg = env_module.GAME_FG_COLOURS
    return (color_bg, color_fg)
