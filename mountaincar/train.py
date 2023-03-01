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
import time
import datetime

from get_reward import *
from DQN import *


def train_dqn(episodes, env, reward_type, param_dict):

    # create timestamp, time marker, and data folder
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    start_time = time.time()
    last_time = time.time()
    path = f'./MC_v3_data/{timestamp}/'
    if not os.path.exists(path):
        os.makedirs(path)

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
            agent.replay(score)

            if i % 50 == 0:
                print("episode: %i/%i, step: %i/%i, score: %.3f, epsilon: %.5f" % (e+1, episodes, i, max_steps, score, agent.epsilon))
            
            if done:
                # testing with done condition
                # if agent.epsilon >= agent.epsilon_min:
                #     agent.epsilon = agent.epsilon_min

                print("episode: %i/%i (reached goal), score: %.3f" % (e+1, episodes, score))
                print('time since start: %is, ' % (time.time() - start_time), end='')
                print('episode length: %is, ' % (time.time() - last_time), end='')
                print('steps taken: %i\n,' % (i))
                last_time = time.time()

                if i < best_steps:
                    best_steps = i
                    agent.save(f'./MC_v3_data/{timestamp}/model_{reward_type}_{timestamp}_{best_steps}.h5')
                break
                
        if not done:
            # testing with not done condition
            # if agent.epsilon <= 0.5:
            #     agent.epsilon = 0.5

            print("episode: %i/%i (did not reach goal), score: %.3f" % (e+1, episodes, score))
            print('time since start: %is, ' % (time.time() - start_time), end='')
            print('episode length: %is\n' % (time.time() - last_time))
            last_time = time.time()

        score_hist.append(score)
        step_count.append(i)

    colors = {"original" : "blue", "plus_velocity" : "green", "human" : "red", "test" : "yellow", "adibyte": "orange"}

    # save plots
    plt.plot([i+1 for i in range(episodes)], score_hist, color = colors[reward_type])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title("Mountain Car Final Score with {} Reward Function".format(reward_type))
    plt.savefig(f'./MC_v3_data/{timestamp}/plot_scores_{reward_type}_{timestamp}.png')
    plt.clf()

    plt.plot([i+1 for i in range(episodes)], step_count, color = colors[reward_type])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title("Mountain Car Steps per Episode with {} Reward Function".format(reward_type))
    plt.savefig(f'./MC_v3_data/{timestamp}/plot_steps_{reward_type}_{timestamp}.png')
    plt.clf()

    # save logs
    total_time = time.time() - start_time

    text = f'Training with {reward_type} reward on {episodes} episodes:\n'
    text = text + f'Best score: {max(score_hist)}\n'
    text = text + f'Minimum steps: {best_steps}\n\n'
    text = text + 'Training time: %is, Avg per episode: %is\n' % (total_time, total_time / episodes)
    text = text + 'Parameters:\n'
    text = text + f"epsilon: {param_dict['epsilon']}\n"
    text = text + f"epsilon_min: {param_dict['epsilon_min']}\n"
    text = text + f"epsilon_decay: {param_dict['epsilon_decay']}\n"
    text = text + f"gamma: {param_dict['gamma']}\n"
    text = text + f"batch_size: {param_dict['batch_size']}\n"
    text = text + f"lr: {param_dict['lr']}\n"
    text = text + f"memory: {param_dict['memory']}\n"
    text = text + f"max_steps: {param_dict['max_steps']}"

    with open(f"./MC_v3_data/{timestamp}/log_{timestamp}.txt", 'w') as f:   
        f.write(text)