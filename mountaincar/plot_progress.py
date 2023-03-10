import numpy as np
import pickle as pkl
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import re
import os

# prompt the user for file path in MC_v3_data to automatically store file in right location
file_path = ""
while re.search("(/MC_v3_data/.+/)(.+).pkl", file_path) is None:
    file_path = input("Please provide the file path for the pickle object to save the plot in the MC_v3_data folder.\n")
    if re.search("(/MC_v3_data/.+/)(.+).pkl", file_path) is None:
        print("Please try again with a pickle object in the MC_v3_data folder.\n")

# extract the file name and folder path within the MC_v3_data folder
x = re.findall("(/MC_v3_data/.+/)(.+).pkl", file_path)
folders = x[0][0]
file_name = x[0][1]

# get pickle data dictionary
with open(f".{folders}/{file_name}.pkl", "rb") as pickle_file:
    pickle_data = pkl.load(pickle_file)
print("Loaded pickle data\n")

# search current folder for model type
model_type = ""
dir_list = os.listdir(path = f".{folders[0:(len(folders)-1)]}")
for file in dir_list:
    if re.search("original", file) is not None:
        model_type = "original"
        break
    elif re.search("plus_velocity", file) is not None:
        model_type = "plus_velocity"
        break
    elif re.search("human", file) is not None:
        model_type = "human"
        break
    elif re.search("test", file) is not None:
        model_type = "test"
        break
    elif re.search("adibyte", file) is not None:
        model_type = "adibyte"
        break

if len(model_type) == 0:
    raise Exception("Cannot find model type!\n")
print(f"Model type: {model_type}")

mpl.rcParams ["figure.figsize"] = [8, 6]

# dictionary of colors for different model types
colors = {"original" : "blue", "plus_velocity" : "green", "human" : "red", "test" : "purple", "adibyte": "orange"}

# score plot
fig, ax = plt.subplots()
ax.plot(pickle_data["score_hist"], color = colors[model_type])
plt.xlabel("Episodes")
plt.ylabel("Score History")
plt.title(f"Mountain Car Score per Episode with {model_type} Reward Function")
fig = plt.gcf()
fig.savefig(f".{folders}/high_res_score_plot.svg", format = "svg")
plt.close()
print("Saved the scores plot\n")

# steps plot
fig, ax = plt.subplots()
ax.plot(pickle_data["step_count"], color = colors[model_type])
plt.xlabel("Episodes")
plt.ylabel("Step History")
plt.title(f"Mountain Car Steps per Episode with {model_type} Reward Function")
fig = plt.gcf()
fig.savefig(f".{folders}/high_res_steps_plot.svg", format = "svg")
plt.close()
print("Saved the steps plot\n")

# best steps plot
# comment out if using a previous run
position = []
velocity = []
for step in pickle_data["best_step_data"]:
    position.append(step[0])
    velocity.append(step[1])
max_velo = max(velocity)
min_velo = min(velocity)
fig, ax = plt.subplots()
ax.plot(position, velocity, color = colors[model_type])
plt.vlines(x = -0.5, ymin = min_velo, ymax = max_velo, color = 'black', linestyle = ':', label = "Bottom of Hill")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title(f"{pickle_data['best_steps']} Steps States Plot with {model_type} Reward Function")
plt.legend(loc = 'lower right')
fig = plt.gcf()
fig.savefig(f".{folders}/high_res_states_plot.svg", format = "svg")
plt.close()
print("Saved the states plot\n")

print("Done.\n")