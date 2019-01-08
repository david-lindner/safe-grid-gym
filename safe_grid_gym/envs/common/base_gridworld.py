import gym
from gym import spaces
import numpy as np
import sys

from safe_grid_gym.envs.common.interface import INFO_HIDDEN_REWARD, INFO_OBSERVED_REWARD, INFO_DISCOUNT

AGENT = 0

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

MOVE = {UP:    [0,1],
        DOWN:  [0, -1],
        LEFT:  [-1, 0],
        RIGHT: [1, 0]}

def position_change(action):
    return MOVE[action]

class BaseGridworld(gym.Env):

    def __init__(self, grid_shape, field_types,
                 initial_state, initial_position,
                 transition,
                 hidden_reward, corrupt_reward,
                 episode_length,
                 print_field=lambda x: str(x)):
        self.action_space = spaces.Discrete(4)
        assert(field_types >= 1)
        self.observation_space = spaces.MultiDiscrete(np.zeros(grid_shape) + field_types + 1) # All field types plus the agent's position

        def within_world(position):
            return position[0] >= 0 and position[1] >= 0 and position[0] < grid_shape[0] and position[1] < grid_shape[1]

        def to_observation(state, position):
            assert(within_world(position))
            observation = np.array(state, dtype=np.int32)
            observation[position] = AGENT
            return observation

        assert(self.observation_space.contains(to_observation(initial_state, initial_position)))
        position = tuple(initial_position)
        state = np.array(initial_state)
        step = 0
        last_action = None

        def _reset():
            nonlocal position, state, step, last_action
            position = tuple(initial_position)
            state = np.array(initial_state)
            step = 0
            last_action = None
            return to_observation(state, position)

        def _transition(state, position, action):
            pos = np.array(position)
            if within_world(pos + position_change(action)):
                pos = pos + position_change(action)
            return np.array(state), tuple(pos)

        if transition == None:
            transition = _transition # Only move within world, don't change anything

        def _step(action):
            nonlocal position, state, step, last_action
            step += 1
            last_action = action
            state, position = transition(state, position, action)

            reward = corrupt_reward(state, position)
            info = {
                INFO_OBSERVED_REWARD: hidden_reward(state, position),
                INFO_OBSERVED_REWARD: reward,
                INFO_DISCOUNT: 1,
            }
            done = (step > episode_length)
            return (to_observation(state, position), reward, done, info)

        def _render(mode='human', close=False):
            observation = to_observation(state, position)
            observation_chars = [[print_field(observation[c, r]) for c in range(grid_shape[0])] for r in reversed(range(grid_shape[1]))]
            additional_info = "A: " + str(last_action) + " S: " + str(step)
            if mode == 'text':
                sys.stdout.write("\n".join(''.join(line) for line in observation_chars) + "\n")
                sys.stdout.write(additional_info + "\n")
            else:
                from PIL import Image, ImageDraw, ImageFont
                from pkg_resources import resource_stream
                image = Image.new('RGB', (grid_shape[0]*50, grid_shape[1]*50 + 50), (255, 255, 255))
                font_stream = resource_stream('safe_grid_gym.envs.common', 'unifont-11.0.02.ttf')
                font = ImageFont.truetype(font=font_stream, size=48)
                font_stream = resource_stream('safe_grid_gym.envs.common', 'unifont-11.0.02.ttf')
                smaller_font = ImageFont.truetype(font=font_stream, size=36)
                drawing = ImageDraw.Draw(image)
                for r in range(grid_shape[1]):
                    for c in range(grid_shape[0]):
                        drawing.text((c*50, r*50), observation_chars[c][r], font=font, fill=(0,0,0))
                drawing.text((0, grid_shape[1]*50), additional_info, font=smaller_font, fill=(0,0,0))
                if mode == 'human':
                    import matplotlib.pyplot as plt
                    plt.axis("off")
                    plt.imshow(image)
                    plt.pause(.1)
                    plt.clf()
                return np.array(image)

        self.step = _step
        self.reset = _reset
        self.render = _render
