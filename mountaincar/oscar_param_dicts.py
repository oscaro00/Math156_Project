param_dict2 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9999, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .001, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

param_dict3 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .997, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .001, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

param_dict4 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .002, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

param_dict5 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9999, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .002, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

param_dict6 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .997, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .002, # try .002
    'memory': 100000,
    'max_steps': 600 # try 1000
}

param_dict7 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995, # try .9999 or .997
    'gamma': .95,
    'batch_size': 64, # try 128
    'lr': .001, # try .002
    'memory': 100000,
    'max_steps': 1000 # try 1000
}

param_dict8 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995, # try .9999 or .997
    'gamma': .95,
    'batch_size': 128, # try 128
    'lr': .001, # try .002
    'memory': 100000,
    'max_steps': 1000 # try 1000
}

param_dict9 = {
    'epsilon': 1.0,
    'epsilon_min': .01,
    'epsilon_decay': .9995, # try .9999 or .997
    'gamma': .95,
    'batch_size': 128, # try 128
    'lr': .002, # try .002
    'memory': 100000,
    'max_steps': 1000 # try 1000
}


# train_dqn(episodes, env, reward_type, param_dict2)
# train_dqn(episodes, env, reward_type, param_dict3)
# train_dqn(episodes, env, reward_type, param_dict4)
# train_dqn(episodes, env, reward_type, param_dict5)
# train_dqn(episodes, env, reward_type, param_dict6)
# train_dqn(episodes, env, reward_type, param_dict7)
# train_dqn(episodes, env, reward_type, param_dict8)
# train_dqn(episodes, env, reward_type, param_dict9)