from gym.envs.registration import register
from ai_safety_gridworlds.helpers.factory import _environment_classes

from safe_grid_gym.envs import GridworldEnv
import safe_grid_gym.envs.toy_grids as _toy_grids

env_list = _environment_classes.keys()


def to_gym_id(env_name):
    result = []
    nextUpper = True
    for char in env_name:
        if nextUpper:
            result.append(char.upper())
            nextUpper = False
        elif char == "_":
            nextUpper = True
        else:
            result.append(char)
    return "".join(result)


for env_name in env_list:
    gym_id_prefix = to_gym_id(str(env_name))
    if gym_id_prefix == "ConveyorBelt":
        for variant in ['vase', 'sushi', 'sushi_goal', 'sushi_goal2']:
            register(
                id=to_gym_id(str(variant)) + "-v0",
                entry_point="safe_grid_gym.envs.gridworlds_env:GridworldEnv",
                kwargs={"env_name": env_name, "variant": variant},
            )
    else:
        register(
            id=gym_id_prefix + "-v0",
            entry_point="safe_grid_gym.envs.gridworlds_env:GridworldEnv",
            kwargs={"env_name": env_name},
        )

register(
    id="TransitionBoatRace-v0",
    entry_point="safe_grid_gym.envs.gridworlds_env:GridworldEnv",
    kwargs={"env_name": "boat_race", "use_transitions": True},
)

register(
    id="ToyGridworldUncorrupted-v0",
    entry_point="safe_grid_gym.envs.common.base_gridworld:BaseGridworld",
    kwargs={
        "grid_shape": _toy_grids.GRID_SHAPE,
        "field_types": 1,
        "initial_state": _toy_grids.INITIAL_STATE,
        "initial_position": _toy_grids.INITIAL_POSITION,
        "transition": None,
        "hidden_reward": _toy_grids.hidden_reward,
        "corrupt_reward": _toy_grids.hidden_reward,
        "episode_length": _toy_grids.EPISODE_LENGTH,
        "print_field": _toy_grids.print_field,
    },
)

register(
    id="ToyGridworldCorners-v0",
    entry_point="safe_grid_gym.envs.common.base_gridworld:BaseGridworld",
    kwargs={
        "grid_shape": _toy_grids.GRID_SHAPE,
        "field_types": 1,
        "initial_state": _toy_grids.INITIAL_STATE,
        "initial_position": _toy_grids.INITIAL_POSITION,
        "transition": None,
        "hidden_reward": _toy_grids.hidden_reward,
        "corrupt_reward": _toy_grids.corrupt_corners,
        "episode_length": _toy_grids.EPISODE_LENGTH,
        "print_field": _toy_grids.print_field,
    },
)

register(
    id="ToyGridworldOnTheWay-v0",
    entry_point="safe_grid_gym.envs.common.base_gridworld:BaseGridworld",
    kwargs={
        "grid_shape": _toy_grids.GRID_SHAPE,
        "field_types": 1,
        "initial_state": _toy_grids.INITIAL_STATE,
        "initial_position": _toy_grids.INITIAL_POSITION,
        "transition": None,
        "hidden_reward": _toy_grids.hidden_reward,
        "corrupt_reward": _toy_grids.corrupt_on_the_way,
        "episode_length": _toy_grids.EPISODE_LENGTH,
        "print_field": _toy_grids.print_field,
    },
)
