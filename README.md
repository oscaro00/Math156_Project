# Math156_Project
 Project Group 9

This repository contains group 9's code for our math 156 machine learning group project.

See our project_proposal.pdf for more details about our model.

## other folder

This folder contains lots of jupyter notebook files used for testing, but it was not a robust method for version control with many people. Note: not all files run properly in this folder as most were used for our initial testing.

## mountaincar folder

This folder contains the .py files that were used for training the deep Q networks.

Each test that was trained to completion creates a data folder with relevant logs, plots, and model objects.

-DQN.py defines the deep q learning class
-MC_v3.py is the main file that trains the models
-functions.py has the various reward functions and other auxiliary functions
-jon_train.py is Jonathn's slightly modified train file
-oscar_param_dicts.py holds hyperparameter variations to use while training
-plot_progress.py creates the score, step, and phase plots given a model
-record_model.py creates a video of the mountain car from the resulting model
-shawn_param.py hold hyperparameter variations to use while training
-train.py has the code for iterating through steps and updating the necessary values
-visualizations.ipynb creates the decision space plots for a given model
