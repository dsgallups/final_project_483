from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import pandas as pd
import numpy as np
from gym import spaces
import pygame
import os
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from environment import TFPong as Pong

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

  try:
    os.environ["DISPLAY"]
  except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.display.set_mode((10, 10), display=0)

env_name = "CartPole-v0"
env = suite_gym.load(env_name)
tf_env = tf_py_environment.TFPyEnvironment(env)
print("Env Current Time Step")
print(env._current_time_step)
pong = Pong()
tf_pong = tf_py_environment.TFPyEnvironment(pong)
print("Pong Current Time Step")
print(pong._current_time_step)