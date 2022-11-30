from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import pandas as pd
import numpy as np
from gym import spaces

env = gym.make('MountainCar-v0')


df = pd.DataFrame([[10, 20, 30], [100, 200, 300]],
                  columns=['foo', 'bar', 'baz'])
def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except Exception:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except Exception:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')

##############################################################################

from tf_agents.environments import gym_wrapper
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.typing import types

from typing import Any, Callable, Dict, Optional, Sequence, Text

TimeLimitWrapperType = Callable[[py_environment.PyEnvironment, int],
                                py_environment.PyEnvironment]

height=800
width=600

gym_env = gym.spec('MountainCar-v0').make()

print(get_methods(gym_env))


print(spaces.Dict({
                "paddle_y": spaces.Box(low=0, high=height, shape=()), 
                "ball_y": spaces.Box(low=0, high=height, shape=()), 
                "diff_x": spaces.Box(low=0, high=width, shape=())
            }))

print(np.array([2, 3, 4]))