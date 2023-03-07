# You may need to run the following commands to get the necessary packages:
# pip install imageio==2.4.1
# apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
# pip install colabgymrender
# pip install opencv-python

# IMPORTANT:

# recorder.py in the colabgymrender package does not work with the current version of colabgymrender!!
# To fix, find recorder.py in your files and delete both instances of mode = "rgb_array".
# Afterwards, do not re-run pip install colabgymrender or you'll have to edit the file again.

import gymnasium as gym
import numpy as np
import keras
from keras.models import load_model
from gym import wrappers

from colabgymrender.recorder import Recorder
import re

# prompt the user for file path in MC_v3_data to automatically store file in right location
file_path = ""
while re.search("(/MC_v3_data/.+/)(.+).h5", file_path) is None:
    file_path = input("Please provide the file path for the model to record in the MC_v3_data folder.\n")
    if re.search("(/MC_v3_data/.+/)(.+).h5", file_path) is None:
        print("Please try again with a model in the MC_v3_data folder.")

# extract the file name and folder path within the MC_v3_data folder
x = re.findall("(/MC_v3_data/.+/)(.+).h5", file_path)
folders = x[0][0]
file_name = x[0][1]

# load the model to record
model = load_model(f".{folders}/{file_name}.h5")

env = gym.make("MountainCar-v0", render_mode = "rgb_array")
# set the directory to the folder of the model file and removes the last /
env = Recorder(env, directory = f'.{folders[0:(len(folders)-1)]}')

# I don't see a way to control the resulting file name, so the file names are just gross numbers unfortunately

# prepare to test the model

scores = 0
episodes = 1

# run one episode without training to show the optimal actions of the model
for e in range(episodes):
    state = env.reset()[0]
    done = False
    time_taken = 0
    while not done:
        env.render()
        #env.capture_frame()
        time_taken += 1
        state = np.reshape(state, [1,2])
        action = model.predict(state)
        action = np.argmax(action)
        next_state, reward, done, info = env.step(action)[0:4]
        state = next_state
    print("time_taken:", time_taken)
env.close()

print("The video file has been saved.")