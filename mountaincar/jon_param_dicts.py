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
    'max_steps': 1000, # try 1000
}

episodes = 100
reward_type = 'adibyte'

train_dqn(episodes, env, reward_type, param_dict, done_condition=[True, [0.1, 0.5]])