# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 10000
REPLAY_START_SIZE = 1000
BATCH_SIZE = 64
GAMMA = 0.95

class DDPG:
    """docstring for DDPG"""
    def __init__(self, environment):
        self.name = 'DDPG' # name for uploading results
        self.environment = environment
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.actor_network = ActorNetwork(state_size = environment.observation_space.shape[0],action_size = environment.action_space.shape[0])
        self.critic_network = CriticNetwork(state_size = environment.observation_space.shape[0],action_size = environment.action_space.shape[0])
        # initialize replay buffer
        self.replay_buffer = deque()

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(environment.action_space.shape[0])

        # Initialize time step
        self.time_step = 0

    def set_init_observation(self,observation):
        # receive initial observation state
        self.state = observation

    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        action_batch = np.resize(action_batch,[BATCH_SIZE,1])

        # Calculate y
        y_batch = []
        next_action_batch = self.actor_network.target_evaluate(next_state_batch)
        q_value_batch = self.critic_network.target_evaluate(next_state_batch,next_action_batch)
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.evaluate(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)/BATCH_SIZE

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def get_action(self):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.get_action(self.state)
        return np.clip(action+self.exploration_noise.noise(),self.environment.action_space.low,self.environment.action_space.high)

    def set_feedback(self,observation,action,reward,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        next_state = observation
        self.replay_buffer.append((self.state,action,reward,next_state,done))
        # Update current state
        self.state = next_state
        # Update time step
        self.time_step += 1

        # Limit the replay buffer size
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer.popleft()

        # Store transitions to replay start size then start training
        if self.time_step >  REPLAY_START_SIZE:
            self.train()

        if self.time_step % 10000 == 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()










