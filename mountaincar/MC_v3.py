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
from datetime import datetime

from DQN import *
from get_reward import *

# options: original, plus_velocity, human
reward_type = ""
while reward_type not in ["original", "plus_velocity", "human"]:
    reward_type = input("What reward function do you want to train on? (original, plus_velocity, human)\n")
    if reward_type not in ["original", "plus_velocity", "human"]:
        print("Please try again.\n")

episodes = 0
while episodes < 1 or episodes % 1 != 0:
    episodes = input("How many episodes do you want to train on? (positive integer)\n")
    episodes = int(episodes)
    if episodes < 1 or episodes % 1 != 0:
        print("Please try again.\n")

# checks that GPU is being used
if len(tf.config.list_physical_devices('GPU')) > 0:
    hardware = 'GPU'
else:
    hardware = "CPU"

response = ""
while response not in ["y", "n"]:
    message = "Do you want to begin the training on {} episodes using {} and the {} reward function? (y/n)\n".format(episodes, hardware, reward_type)
    response = input(message)
    if response not in ["y", "n"]:
        print("Please try again.\n")


if response == 'n':
    raise Exception("Incorrect training conditions.")


## Beginning the training code below

curr_time = datetime.now()
time_stamp = curr_time.timestamp()
date_time = datetime.fromtimestamp(time_stamp)

date = str(date_time)[0:10]
time = str(date_time)[11:19].replace(":", '-') 


env = gym.make('MountainCar-v0', render_mode = "human")
#env.seed(134)
np.random.seed(458)




print(env.observation_space)
print(env.action_space)

loss, step_count = train_dqn(episodes, env, reward_type, date, time)



colors = {"original" : "blue", "plus_velocity" : "green", "human" : "red"}

plt.plot([i+1 for i in range(episodes)], loss, color = colors[reward_type])
plt.ylabel('Score per Episode')
plt.title("Mountain Car Final Score with {} Reward Function".format(reward_type))
plt.savefig("./MC_v3_plots/loss_{}_{}_{}.png".format(reward_type, date, time))
plt.show()



plt.plot([i+1 for i in range(episodes)], step_count, color = colors[reward_type])
plt.ylabel('Score per Episode')
plt.title("Mountain Car Final Score with {} Reward Function".format(reward_type))
plt.savefig("MC_v3_plots/step_{}_{}_{}.png".format(reward_type, date, time))
plt.show()