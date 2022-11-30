from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import reverb


from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
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

import pygame
from environment import TFPong as Pong
import time
#Hyperparameters
num_iterations = 50 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1000 # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # For Adam optimizer
log_interval = 3  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 40  # @param {type:"integer"}


if __name__ == "__main__":
    print("------------------------------------------------------------")
    print("STARTING PONG")
    print("------------------------------------------------------------")

    env = Pong(show_display=True, title="Env")
    train_py_env = Pong(show_display=True, title="Train Py Env")
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_py_env = Pong(show_display=True, title="Eval Py Env")
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    

    utils.validate_py_environment(env, episodes=5)

    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)


    print('Action Spec:')
    print(env.action_spec())

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    print("Action Tensor Spec:")
    print(action_tensor_spec)
    
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    print ("Number of Actions:")
    print(num_actions)

    #Create dense layers with relu activation and a normal distribution (truncated)
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            #this sets the initial weights of each unit
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            )
        )

    fc_layer_params = (4, 3)
    #so then create the dense layers based on our fc_layer_params var
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    
    #this creates 1 q_value per available action as output
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation= None,
        
        #no idea why intiialize this as uniform between -.03 and .03
        kernel_initializer = tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),

        #also no idea why we give them initial biases of -.2
        bias_initializer=tf.keras.initializers.Constant(-.2)
        )
    
    #this does not include the input layer I believe
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    #create the optimizer, which is adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)


    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    print("Agent Collect Data Spec")
    print(agent.collect_data_spec)


    #Policy used for evaluation and deployment
    eval_policy = agent.policy
    

    #policy used for data collection
    collect_policy = agent.collect_policy
    


    #Create a random policy for baselining
    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(),
        train_env.action_spec()
    )


    #Compute average return for the environment and policy
    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0

        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    print("------------------------------------------------------------")
    print("Starting Random Policy Evaluation")
    start = time.time()
    avg_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    end = time.time()
    print("Average Return")
    print(avg_return)
    print ("Completed in {} seconds".format(end - start))
    print("------------------------------------------------------------")


    #Create a replay buffer
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        #this adds a row
        sequence_length=2,
        local_server=reverb_server
    )

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2
    )

    #We will want to make our own driver if we are to train two agents togther in the future.

    # final_state = 
    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True
        ),
        [rb_observer],
        max_steps=initial_collect_steps
    ).run(train_py_env.reset())
    
    
    print("\nCollection of Trajectories: ")
    print(iter(replay_buffer.as_dataset()).next())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    dataset

    iterator = iter(dataset)
    print("\nITERATOR -------------------------------")
    print(iterator)


    # print("Comparison (CAUTION, LONG OUTPUT)")
    # print(iterator.next())


    #Now, train the agent

    agent.train = common.function(agent.train)

    #reset train step
    agent.train_step_counter.assign(0)

    #evaluate the agent policy prior to training
    print("------------------------------------------------------------")
    print("Conducting initial agent evaluation:")
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    print("Finished.")
    print("------------------------------------------------------------")


    #reset env
    time_step = train_py_env.reset()

    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True
        ),
        [rb_observer],
        max_steps=collect_steps_per_iteration
    )
    print("------------------------------------------------------------")
    print("Now training model...")
    print("------------------------------------------------------------")
    for _ in range(num_iterations):

        #run the driver for the number of iterations identified
        # print("here0")
        time_step, _ = collect_driver.run(time_step)
        # print("here1")
        experience, unused_info = next(iterator)
        # print("here2")
        train_loss = agent.train(experience).loss

        #do some logging
        step = agent.train_step_counter.numpy()
        # print(step)

        if step % log_interval == 0:
            print("step = {0}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            print("------------------------------------------------------------")
            print("Evaluating Agent")
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)
            print("------------------------------------------------------------")




    print("------------------------------------------------------------")
    print("ENDING PONG")
    print("------------------------------------------------------------")

