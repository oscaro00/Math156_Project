from train import* 
from functions import*
from DQN import*
# from MC_v3 import*

env = gym.make('MountainCar-v0', render_mode = "human")
np.random.seed(458)

param_dict1 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9999, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .001, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

episodes = 100
reward_type="orginal"

train_dqn(episodes, env, reward_type, param_dict1)