from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import pygame
from environment import TFPong as Pong

if __name__ == "__main__":
    print("------------------------------------------------------------")
    print("STARTING PONG")
    print("------------------------------------------------------------")

    env = Pong()
    utils.validate_py_environment(env, episodes=5)
    print("finished")
