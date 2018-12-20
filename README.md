# safe-grid-gym

An [OpenAI Gym](https://github.com/openai/gym) interface for the [AI safety gridworlds by DeepMind](https://github.com/deepmind/ai-safety-gridworlds), which are implemented in [https://github.com/deepmind/pycolab](pycolab).


This repository combines and extends two previous implementations which can be found at:
   - https://github.com/n0p2/gym_ai_safety_gridworlds
   - https://github.com/n0p2/ai-safety-gridworlds-viewer

## Features

`safe_grid_gym` additionally provides:
   - Additional features for the Gym environment:
      - A parameter that can be set to get the true hidden reward from the gridworld environments. This allows to test agents on the hidden reward as well as the observed reward.
      - Additional render modes `ansi` and `rgb_array` allowing for more automated experimentation
   - Easier dependency management by providing a `setup.py`
   - Unittests for the Gym environment using the demonstrations provided by in the `ai-safety-gridworlds` repository

To handle the dependency on the `ai-safety-gridworlds` we use a [fork of the official repository](https://github.com/jvmancuso/ai-safety-gridworlds) that provides a `setup.py`.

You can use the code from the official `ai-safety-gridworlds` repository instead by adding it to your `PYTHONPATH`.

## Usage

By using `safe_grid_gym` the AI safety gridworlds can by used like any other gym environment. For example to take 10 random actions in the boat race environment and render the gridworld, you can do:

```python
from safe_grid_gym.envs import GridworldEnv

env = GridworldEnv("boat_race")
action_space = env.action_space

for i in range(10):
   action = action_space.sample()
   state, reward, done, info = env.step(action)
   env.render(mode="human")
```
