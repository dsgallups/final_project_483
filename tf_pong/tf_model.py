from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

"""
Args:
  time_step_spec: A TimeStep spec of the expected time_steps.
  action_spec: A nest of BoundedTensorSpec representing the actions.
  q_network: A tf_agents.network.Network to be used by the agent. The
    network will be called with call(observation, step_type) and should emit logits over the action space.
  optimizer: The optimizer to use for training.
  observation_and_action_constraint_splitter: A function used to process
    observations with action constraints. These constraints can indicate, for example, a mask of valid/invalid actions for a given state of the environment. The function takes in a full observation and returns a tuple consisting of 1) the part of the observation intended as input to the network and 2) the constraint. An example observation_and_action_constraint_splitter could be as simple as: def observation\_and\_action\_constraint\_splitter(observation): &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return observation\['network\_input'\], observation\['constraint'\] &nbsp;&nbsp;&nbsp;&nbsp;     *Note*: when using observation_and_action_constraint_splitter, make sure the provided q_network is compatible with the network-specific half of the output of the observation_and_action_constraint_splitter. In particular, observation_and_action_constraint_splitter will be called on the observation before passing to the network. If observation_and_action_constraint_splitter is None, action constraints are not applied.
  epsilon_greedy: probability of choosing a random action in the default
    epsilon-greedy collect policy (used only if a wrapper is not provided to the collect_policy method). Only one of epsilon_greedy and boltzmann_temperature should be provided.
  n_step_update: The number of steps to consider when computing TD error and
    TD loss. Defaults to single-step updates. Note that this requires the user to call train on Trajectory objects with a time dimension of n_step_update + 1. However, note that we do not yet support n_step_update > 1 in the case of RNNs (i.e., non-empty q_network.state_spec).
  boltzmann_temperature: Temperature value to use for Boltzmann sampling of
    the actions during data collection. The closer to 0.0, the higher the probability of choosing the best action. Only one of epsilon_greedy and boltzmann_temperature should be provided.
  emit_log_probability: Whether policies emit log probabilities or not.
  target_q_network: (Optional.) A tf_agents.network.Network
    to be used as the target network during Q learning. Every target_update_period train steps, the weights from q_network are copied (possibly with smoothing via target_update_tau) to target_q_network.

    If target_q_network is not provided, it is created by making a copy of q_network, which initializes a new network with the same structure and its own layers and weights.

    Network copying is performed via the Network.copy superclass method, and may inadvertently lead to the resulting network to share weights with the original. This can happen if, for example, the original network accepted a pre-built Keras layer in its __init__, or accepted a Keras layer that wasn't built, but neglected to create a new copy.

    In these cases, it is up to you to provide a target Network having weights that are not shared with the original q_network. If you provide a target_q_network that shares any weights with q_network, a warning will be logged but no exception is thrown.

    Note; shallow copies of Keras layers may be built via the code:

    python new\_layer = type(layer).from\_config(layer.get\_config())   target_update_tau: Factor for soft update of the target networks.
  target_update_period: Period for soft update of the target networks.
  td_errors_loss_fn: A function for computing the TD errors loss. If None, a
    default value of element_wise_huber_loss is used. This function takes as input the target and the estimated Q values and returns the loss for each element of the batch.
  gamma: A discount factor for future rewards.
  reward_scale_factor: Multiplicative scale for the reward.
  gradient_clipping: Norm length to clip gradients.
  debug_summaries: A bool to gather debug summaries.
  summarize_grads_and_vars: If True, gradient and network variable summaries
    will be written during training.
  train_step_counter: An optional counter to increment every time the train
    op is run. Defaults to the global_step.
  training_data_spec: A nest of TensorSpec specifying the structure of data
    the train() function expects. If None, defaults to the trajectory_spec of the collect_policy.
  name: The name of this agent. All variables in this module will fall
    under that name. Defaults to the class name.

Raises:
  ValueError: If action_spec contains more than one action or action
    spec minimum is not equal to 0.
  ValueError: If the q networks do not emit floating point outputs with
    inner shape matching action_spec.
  NotImplementedError: If q_network has non-empty state_spec (i.e., an
    RNN is provided) and n_step_update > 1.



Several things to mention:
    DqnAgent requests expected time steps...but our game could go on for quite a while.
    The timestep contains an observation from the environment, uses TimeStep spec
"""

class PongModel:

    def __init__(self):
        self.agent = dqn_agent.DqnAgent()


    def get_decision(self, paddle_x, paddle_y, ball_x, ball_y):
        step = tf_agents.trajectories.TimeStep(tf_agents.trajectories.StepType.FIRST)
        pass



