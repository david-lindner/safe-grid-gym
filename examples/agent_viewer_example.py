"""
Example demonstrating the usage of the AgentViewer independant of the gym
interface.

This was taken from https://github.com/n0p2/ai-safety-gridworlds-viewer
"""

import argparse
import logging
import importlib
import time
import numpy as np

from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.helpers import factory

from safe_grid_gym.viewer import AgentViewer, display


# --------
# test view_agent.AgentViewer
# --------


def view_agent(args):
    environment_name = args.environment_name

    # Color mapppings (foreground and background) are required
    # to initialize an AgentViewer. (Somewhat supperisingly)
    # such mappings are not stored in an instance of an env class.
    # Each env module does include these mappings.
    (color_bg, color_fg) = get_color_map(environment_name)

    create_av = lambda: AgentViewer(args.pause, color_bg=color_bg, color_fg=color_fg)
    if args.context:
        with create_av() as av:
            view_agent_env(av, args)
    else:
        av = create_av()
        view_agent_env(av, args)
        del av


def view_agent_env(av, args):
    logger = get_logger()

    env = factory.get_environment_obj(args.environment_name)
    env.reset()
    av.display(env)
    episode_return = 0

    actions = get_actions(args)
    e = 0
    for (i, action) in enumerate(actions):
        timestep = env.step(action)
        episode_return += reward(timestep)
        av.display(env)
        if timestep.step_type.last():
            logger.info("episode %d: %.2f" % (e, episode_return))
            env.reset()
            av.reset_time()
            av.display(env)
            episode_return = 0
            e += 1


def reward(timestep):
    return 0.0 if timestep.reward is None else timestep.reward


def get_color_map(environment_name):
    env_module_name = "ai_safety_gridworlds.environments." + environment_name
    env_module = importlib.import_module(env_module_name)
    color_bg = env_module.GAME_BG_COLOURS
    color_fg = env_module.GAME_FG_COLOURS
    return (color_bg, color_fg)


# --------
# actions
# --------

_actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN, Actions.QUIT]


def get_actions(args):
    if args.rand_act:
        return rand_actions(args.seed, args.steps)
    else:
        demo = demonstrations.get_demonstrations(args.environment_name)[0]
        np.random.seed(demo.seed)
        return demo.actions


def rand_actions(seed=0, steps=10):
    np.random.seed(seed)
    # Actions.QUIT is never chosen in this case
    actions = np.random.randint(0, 4, steps)
    return map(lambda a: _actions[a], actions)


# --------
# test view_agent.display
#
# this include a list of carefuly chosen board values board_*
# --------


board_1 = np.array(
    [
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
        [35, 65, 32, 76, 76, 76, 32, 71, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 76, 76, 76, 32, 32, 35],
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
    ]
)

board_2 = np.array(
    [
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
        [35, 32, 32, 76, 76, 76, 32, 71, 35],
        [35, 65, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 76, 76, 76, 32, 32, 35],
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
    ]
)

board_3 = np.array(
    [
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
        [35, 32, 32, 76, 76, 76, 32, 71, 35],
        [35, 32, 65, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 76, 76, 76, 32, 32, 35],
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
    ]
)

board_4 = np.array(
    [
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
        [35, 32, 32, 76, 76, 76, 32, 71, 35],
        [35, 32, 32, 65, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 32, 32, 35],
        [35, 32, 32, 76, 76, 76, 32, 32, 35],
        [35, 35, 35, 35, 35, 35, 35, 35, 35],
    ]
)

boards = [board_1, board_2, board_3, board_4]


def test_display(pause):
    (color_bg, color_fg) = get_color_map("distributional_shift")
    av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)

    for (i, b) in enumerate(boards):
        display(av._screen, b, 5, 1, av._colour_pair)
        time.sleep(pause)

    del av


# --------
# logging and debugging
# --------
_logger = None


def get_logger():
    """
  singleton
  """
    global _logger
    if _logger is None:
        _logger = logging.getLogger(__file__)
        hdlr = logging.FileHandler(__file__ + ".log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        _logger.addHandler(hdlr)
        _logger.setLevel(logging.DEBUG)

    return _logger


# --------
# main io
# --------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment_name",
        default="distributional_shift",
        help="Environment name as defined in Deepmind AI safety gridworlds",
    )
    parser.add_argument("-p", "--pause", type=float, default=0.05)
    parser.add_argument("-r", "--rand_act", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("-c", "--context", action="store_true")
    parser.add_argument(
        "-t",
        "--test",
        choices=["av", "display"],
        default="av",
        help="Test modules: AgentViwer (av) or display",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test == "av":
        view_agent(args)
    elif args.test == "display":
        test_display(args.pause)
    else:
        raise Exception()
