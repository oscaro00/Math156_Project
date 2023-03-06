from train import *

env = gym.make('MountainCar-v0', render_mode = "human")
np.random.seed(458)

param_dict = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995,# try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .002, # try .002
    'memory': 100000,
    'max_steps': 600, # try 1000
}

episodes = 1
reward_type = 'adibyte'

# train_dqn(episodes, env, reward_type, param_dict, done_condition=[False, [0.3, 0.5]])
train_dqn(episodes, env, reward_type, param_dict, done_condition=(True, [0.3, 0.9]))