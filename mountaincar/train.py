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
import pickle as pkl
import datetime

from functions import *
from DQN import *


def train_dqn(episodes, env, reward_type, param_dict, done_condition=(False, [])):

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
    best_step_data = []

    # initialize agent
    agent = DQN(env.action_space.n, env.observation_space.shape[0], param_dict)

    for e in range(episodes):
        # initialize empty array to hold position and velocity data
        step_data = []
        state = env.reset()[0] # added [0] to drop unnecessary dictionary
        # state = get_state(state) # uncomment to use buckets
        state = np.reshape(state, (1, 2)) # reshape and return bucket index
        score = 0
        max_steps = param_dict['max_steps'] # changed from 1000

        for i in range(max_steps):
            # add position and velocity data to step_data array
            step_data.append([state[0][0],state[0][1]])

            # choose A from Q via eps-greedy policy
            action = agent.act(state)

            # observe R, S' from environment
            next_state, reward, done, _ = env.step(action)[0:4] # added [0:4] to drop unnecessary dictionary
            # next_state = get_state(next_state) # uncomment to use buckets

            # generate custom reward
            reward = get_reward(state, next_state, reward_type, i, max_steps)
            score += reward
            next_state = np.reshape(next_state, (1, 2)) # reshape and return bucket index

            # store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # one step of Q via Bellman's eqn
            agent.replay()

            # update state
            state = next_state

            if i % 50 == 0:
                print("episode: %i/%i, step: %i/%i, score: %.3f, epsilon: %.5f" % (e+1, episodes, i, max_steps, score, agent.epsilon))
            
            if done:
                if done_condition[0]:
                    # testing with done condition
                    if agent.epsilon >= done_condition[1][0]:#agent.epsilon_min:
                        agent.epsilon = done_condition[1][0]#agent.epsilon_min

                print("episode: %i/%i (reached goal), score: %.3f" % (e+1, episodes, score))
                print('time since start: %is, ' % (time.time() - start_time), end='')
                print('episode length: %is, ' % (time.time() - last_time), end='')
                print('steps taken: %i\n' % (i))
                last_time = time.time()

                if i < best_steps:
                    best_steps = i
                    best_step_data = step_data
                    agent.save(f'./MC_v3_data/{timestamp}/model_{reward_type}_{timestamp}_{best_steps}.h5')
                break
                
        if not done:
            if done_condition[0]:
                # testing with not done condition
                if agent.epsilon <= done_condition[1][1]:
                    agent.epsilon = done_condition[1][1]

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

    # save info arrays
    arr_dict = {'score_hist': score_hist, 'step_count': step_count, 'best_steps': best_steps, 'best_step_data': best_step_data}

    with open(f'./MC_v3_data/{timestamp}/arr_{timestamp}.pkl', 'wb') as fp:
        pkl.dump(arr_dict, fp)

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
    text = text + f"max_steps: {param_dict['max_steps']}\n"
    if done_condition[0]:
        text = text + f"done_condition: [{done_condition[1][0]},{done_condition[1][1]}]"

    with open(f'./MC_v3_data/{timestamp}/log_{timestamp}.txt', 'w') as f:   
        f.write(text)