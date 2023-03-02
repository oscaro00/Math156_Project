import gymnasium as gym
import numpy as np
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import tensorflow as tf

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space, param_dict):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = param_dict['epsilon']
        self.epsilon_min = param_dict['epsilon_min']
        self.epsilon_decay = param_dict['epsilon_decay']

        self.gamma = param_dict['gamma']
        self.lr = param_dict['lr']
        self.batch_size = param_dict['batch_size']
        self.memory = deque(maxlen=param_dict['memory'])

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_space, activation=relu))
        model.add(Dense(32, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        # model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def save(self, name):
        self.model.save(name)