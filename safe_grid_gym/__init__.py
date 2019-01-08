
from gym.envs.registration import register

from safe_grid_gym.envs import GridworldEnv

env_list = [
    "friend_foe",
    "conveyor_belt",
    "boat_race",
    "safe_interruptibility",
    "island_navigation",
    "distributional_shift",
    "side_effects_sokoban",
    "absent_supervisor",
    "tomato_watering",
    "tomato_crmdp",
    "whisky_gold",
]

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
    gym_id_prefix = to_gym_id(env_name)
    register(
            id=gym_id_prefix + '-v0',
            entry_point='safe_grid_gym.envs.gridworlds_env:GridworldEnv',
            kwargs={"env_name":env_name,
                    "cheat":False,
            },
    )
