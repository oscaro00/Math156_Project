import numpy as np

def get_reward(state, next_state, reward_type):

    if reward_type == "original":
        reward = 0
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            reward += 10
        else:
            reward -= 0
        if next_state[0] > -0.4:
            reward += (1+next_state[0])**2

        return reward
    
    elif reward_type == "plus_velocity":
        reward = 0
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            reward += 10
        else: 
            reward -= 1
        # if the next action goes higher or has greater speed, reward
        if next_state[0] > state[0][0] or abs(next_state[1]) > abs(state[0][1]):
            reward += 1
            
        return reward 
    
    elif reward_type == "human":
        reward = 0
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            reward += 10
        else:
            reward -= 1
        # if slowing down and going higher right, reward
        if next_state[0] > state[0][0] and next_state[0] > -0.525 and next_state[1] < state[0][1]:
            reward += 2
        # if slowing down and going higher left, reward
        if next_state[0] < state[0][0] and next_state[0] < -0.525 and next_state[1] > state[0][1]:
            reward += 1
        # if speeding up right and going lower left, reward
        if next_state[0] > state[0][0] and next_state[0] < -0.525 and next_state[1] > state[0][1]:
            reward += 1
        # if speeding up left and going lower right, reward
        if next_state[0] < state[0][0] and next_state[0] > -0.525 and next_state[1] < state[0][1]:
            reward += 1

        return reward

    elif reward_type == "test": #leave this space for further test mode
        position=state[0][0]
        velocity=state[0][1]
        
        reward=0
        if position >= 0.5:
            reward=100
        else:
            import math
            #use tanh to map the value of position&veloicty to [-1,1]
            reward=-0.2 + 2*(math.tanh(abs(velocity*10)+position)) #it currently performs very bad 
        return reward

    elif reward_type == "adibyte":
        reward = 0

        if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
            reward += 1.5
        elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
            reward += 1.5
        
        # give more reward if the cart reaches the flag in 200 steps
        if next_state[0] >= 0.5:
            reward += 1000
        # give more reward if cart is near flag
        # elif next_state[0] >= 0.3:
        #     reward += 100
        # elif next_state[0] >= 0.1:
        #     reward += 20
        else:
            # put a penalty if the no of time steps is more
            reward -= 1

        # once the number of steps decreases a lot, penalize time more
        # if best_steps <= max_steps / 2:
        #     reward -= 5

        return reward

    else:
        return 0

# create buckets for the observation space
pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)

def get_state(observation):
    pos, vel = observation
    # pos_bin = int(np.digitize(pos, pos_space))
    # vel_bin = int(np.digitize(vel, vel_space))
    pos_bin = pos_space[np.digitize(pos, pos_space)]
    vel_bin = vel_space[np.digitize(vel, vel_space)]

    return (pos_bin, vel_bin)
