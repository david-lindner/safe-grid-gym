import gym
from gym import spaces
import numpy as np
import sys

from safe_grid_gym.envs.common.interface import (
    INFO_HIDDEN_REWARD,
    INFO_OBSERVED_REWARD,
    INFO_DISCOUNT,
)

AGENT = 0

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

MOVE = {UP: [0, 1], DOWN: [0, -1], LEFT: [-1, 0], RIGHT: [1, 0]}


def position_change(action):
    return MOVE[action]


class BaseGridworld(gym.Env):
    def __init__(
        self,
        grid_shape,
        field_types,
        initial_state,
        initial_position,
        transition,
        hidden_reward,
        corrupt_reward,
        episode_length,
        print_field=lambda x: str(x),
    ):
        self.action_space = spaces.Discrete(4)
        assert field_types >= 1
        self.observation_space = spaces.MultiDiscrete(
            np.zeros(grid_shape) + field_types + 1
        )  # All field types plus the agent's position

        self.grid_shape = grid_shape
        self.field_types = field_types
        self.initial_state = initial_state
        self.initial_position = initial_position
        if transition == None:
            self.transition = (
                self._transition
            )  # Only move within world, don't change anything
        else:
            self.transition = transition
        self._hidden_reward = hidden_reward
        self._corrupt_reward = corrupt_reward
        self.episode_length = episode_length
        self.print_field = print_field

        assert self.observation_space.contains(
            self.to_observation(initial_state, initial_position)
        )
        self.position = tuple(initial_position)
        self.state = np.array(initial_state)
        self.timestep = 0
        self.last_action = None
        self._episode_return = 0.0
        self._hidden_return = 0.0
        self._last_performance = None
        self._reset_next = False

    def _within_world(self, position):
        return (
            position[0] >= 0
            and position[1] >= 0
            and position[0] < self.grid_shape[0]
            and position[1] < self.grid_shape[1]
        )

    def to_observation(self, state, position, dtype=np.float32):
        assert self._within_world(position)
        observation = np.array(state, dtype=dtype)
        observation[position] = AGENT
        return observation

    def reset(self):
        self.position = tuple(self.initial_position)
        self.state = np.array(self.initial_state)
        self.timestep = 0
        self.last_action = None
        self._episode_return = 0.0
        self._hidden_return = 0.0
        self._reset_next = False
        return self.to_observation(self.state, self.position)

    def _transition(self, state, position, action):
        pos = np.array(position)
        if self._within_world(pos + position_change(action)):
            pos = pos + position_change(action)
        return np.array(state), tuple(pos)

    def step(self, action):
        self.timestep += 1
        self.last_action = action
        self.state, self.position = self.transition(self.state, self.position, action)

        reward = self._corrupt_reward(self.state, self.position)
        hidden = self._hidden_reward(self.state, self.position)
        self._episode_return += reward
        self._hidden_return += hidden

        info = {
            INFO_HIDDEN_REWARD: hidden,
            INFO_OBSERVED_REWARD: reward,
            INFO_DISCOUNT: 1,
        }

        done = self.timestep >= self.episode_length
        if done:
            if self._reset_next:
                self._episode_return -= reward  # FIXME: ugly hack
                self.timestep -= 1
                raise RuntimeError("Failed to reset after end of episode.")
            self._last_performance = self._hidden_return
            self._reset_next = True
        obs = self.to_observation(self.state, self.position)

        return obs, reward, done, info

    @property
    def episode_return(self):
        return self._episode_return

    def get_last_performance(self):
        return self._last_performance

    def render(self, mode="human", close=False):
        observation = self.to_observation(self.state, self.position)
        observation_chars = [
            [self.print_field(observation[c, r]) for c in range(self.grid_shape[0])]
            for r in reversed(range(self.grid_shape[1]))
        ]
        additional_info = "A: " + str(self.last_action) + " S: " + str(self.timestep)
        if mode == "text":
            sys.stdout.write(
                "\n".join("".join(line) for line in observation_chars) + "\n"
            )
            sys.stdout.write(additional_info + "\n")
        else:
            from PIL import Image, ImageDraw, ImageFont
            from pkg_resources import resource_stream

            image = Image.new(
                "RGB",
                (self.grid_shape[0] * 50, self.grid_shape[1] * 50 + 50),
                (255, 255, 255),
            )
            font_stream = resource_stream(
                "safe_grid_gym.envs.common", "unifont-11.0.02.ttf"
            )
            font = ImageFont.truetype(font=font_stream, size=48)
            font_stream = resource_stream(
                "safe_grid_gym.envs.common", "unifont-11.0.02.ttf"
            )
            smaller_font = ImageFont.truetype(font=font_stream, size=36)
            drawing = ImageDraw.Draw(image)
            for r in range(self.grid_shape[1]):
                for c in range(self.grid_shape[0]):
                    drawing.text(
                        (c * 50, r * 50),
                        observation_chars[c][r],
                        font=font,
                        fill=(0, 0, 0),
                    )
            drawing.text(
                (0, self.grid_shape[1] * 50),
                additional_info,
                font=smaller_font,
                fill=(0, 0, 0),
            )
            if mode == "human":
                import matplotlib.pyplot as plt

                plt.axis("off")
                plt.imshow(image)
                plt.pause(0.1)
                plt.clf()
            return np.array(image)
