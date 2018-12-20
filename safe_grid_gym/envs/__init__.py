from gym.envs.registration import register
from ai_safety_gridworlds.helpers import factory
from .gridworlds_env import GridworldEnv

entry_point = "safe_grid_gym.envs:GridworldEnv"
env_names = list(factory._environment_classes.keys())


def get_id(env_name):
    return "ai_safety_gridworlds-" + env_name + "-v0"


for env_name in env_names:
    register(
        id=get_id(env_name),
        entry_point=entry_point,
        kwargs={"env_name": env_name, "pause": 0.2},
    )
