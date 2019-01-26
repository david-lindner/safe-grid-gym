# safe-grid-gym

An [OpenAI Gym](https://github.com/openai/gym) interface for the [AI safety gridworlds by DeepMind](https://github.com/deepmind/ai-safety-gridworlds), which are implemented in [pycolab](https://github.com/deepmind/pycolab).


This repository combines and extends two previous implementations which can be found at:
   - https://github.com/n0p2/gym_ai_safety_gridworlds
   - https://github.com/n0p2/ai-safety-gridworlds-viewer

## Features

`safe_grid_gym` additionally provides:
   - Additional toy environments
   - Additional features for the Gym environment:
      - Additional render modes `ansi` and `rgb_array` allowing for more automated experimentation
      - A `TransitionBoatRace` environments which provides the last two boards as state information
   - Easier dependency management by providing a `setup.py`
   - Unittests for the Gym environment using the demonstrations provided by in the `ai-safety-gridworlds` repository

To handle the dependency on the `ai-safety-gridworlds` we use a [fork of the official repository](https://github.com/jvmancuso/ai-safety-gridworlds) that provides a `setup.py`.

You can use the code from the official `ai-safety-gridworlds` repository instead by adding it to your `PYTHONPATH`.

## Usage

By using `safe_grid_gym` the AI safety gridworlds can by used like any other gym environment. For example to take 10 random actions in the boat race environment and render the gridworld, you can do:

```python
import safe_grid_gym
import gym

env = gym.make("BoatRace-v0")
action_space = env.action_space

for i in range(10):
   action = action_space.sample()
   state, reward, done, info = env.step(action)
   env.render(mode="human")
```
