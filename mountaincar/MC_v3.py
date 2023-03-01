from train import *

# ask user for reward type. options: original, plus_velocity, human
reward_type = ""
while reward_type not in ["original", "plus_velocity", "human", "test", "adibyte"]:
    reward_type = input("What reward function do you want to train on? (original, plus_velocity, human, test, adibyte)\n")
    if reward_type not in ["original", "plus_velocity", "human", "test", "adibyte"]:
        print("Please try again.\n")

# ask user for number of episodes
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

env = gym.make('MountainCar-v0', render_mode = "human")
np.random.seed(458)

param_dict = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995,
    'gamma': .95,
    'batch_size': 64,
    'lr': .001,
    'memory': 100000,
    'max_steps': 500
}

train_dqn(episodes, env, reward_type, param_dict)