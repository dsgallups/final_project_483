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
from .game import Game

class TFPong(py_environment.PyEnvironment):

    LEFT, RIGHT = 0, 1

    # One-time setup
    def __init__(self, window=0, width=700, height=500, agent_count=1, show_display=False):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0.0, name='observation'
        )
        


        #to be possibly used in the future...for now 
        #we will train a single agent against our neat AI.
        if window == 0:
            window = pygame.display.set_mode((width, height))
    
        self.show_display = show_display
        self.game = Game(window, width, height)
        self.agent_count = agent_count


        self._state = self._get_state
        self._episode_ended = False
    
    #only call when it is known that the state has been updated. otherwise just grab from state
    def _get_state(self):
        # return three parameters: paddle_y, ball_y and distance between paddle_x and ball_x
        paddle_y = self.game.right_paddle.y
        ball_y = self.game.ball.y
        diff_x = abs(self.game.right_paddle.x - self.game.ball.x)
        return np.array([paddle_y, ball_y, diff_x], dtype=np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game.reset()
        self._state = self._get_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # last action ended "episode", so ignore
            # any current action and start a new episode.
            return self.reset()
        
        if action == 0:
            pass
        elif action == 1:
            self.game.move_paddle(False, True)
        elif action == 2:
            self.game.move_paddle(False, False)

        action_result = self.game.loop()
        self._state = self._get_state()

        if action_result.round_over:
            reward = action_result.right_hits
            return ts.termination(self._state, reward)
        else:
            return ts.transition(
                self._state, reward=0, discount=1.0
            )


        #since the one being trained is on the right, that's how we calculate the reward