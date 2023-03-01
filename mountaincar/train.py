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

import os
import datetime

from get_reward import *
from DQN import *


def train_dqn(episodes, env, reward_type, param_dict):

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")

    score_hist = []
    step_count = []
    best_steps = np.inf

    agent = DQN(env.action_space.n, env.observation_space.shape[0], param_dict)

    for e in range(episodes):
        state = env.reset()[0] # added [0]
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = param_dict['max_steps'] # changed from 1000

        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)[0:4] # added [0:4]
            reward = get_reward(state, next_state, reward_type)
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

            if i % 50 == 0:
                print("episode: {}/{}, step: {}/{}, score:{}".format(e+1, episodes, i, max_steps, score))
            
            if done:
                print("episode: {}/{} (reached goal), score: {}".format(e+1, episodes, score))
                if i < best_steps:
                    agent.save(f'./MC_v3_models/{timestamp}/model_{reward_type}_{timestamp}_{best_steps}.h5')
                break
                
        if not done:
            print("episode: {}/{} (did not reach goal), score: {}".format(e+1, episodes, score))

        score_hist.append(score)
        step_count.append(i)

    path1 = f'./MC_v3_models/{timestamp}/'
    path2 = f'./MC_v3_plots/{timestamp}/'
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)

    colors = {"original" : "blue", "plus_velocity" : "green", "human" : "red"}

    plt.plot([i+1 for i in range(episodes)], score_hist, color = colors[reward_type])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title("Mountain Car Final Score with {} Reward Function".format(reward_type))
    plt.savefig(f'./MC_v3_plots/{timestamp}/scores_{reward_type}_{timestamp}.png')
    plt.clf()

    plt.plot([i+1 for i in range(episodes)], step_count, color = colors[reward_type])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title("Mountain Car Steps per Episode with {} Reward Function".format(reward_type))
    plt.savefig(f'./MC_v3_plots/{timestamp}/steps_{reward_type}_{timestamp}.png')
    plt.clf()