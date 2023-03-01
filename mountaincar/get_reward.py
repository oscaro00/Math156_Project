def get_reward(state, next_state, reward_type, ):

    if reward_type == "original":
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            return 10
        if next_state[0] > -0.4:
            return (1+next_state[0])**2
    
    elif reward_type == "plus_velocity":
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            return 10
        # if the next action goes higher or has greater speed, reward
        if next_state[0] > state[0][0] or abs(next_state[1]) > abs(state[0][1]):
            return 1
        else: 
            return 0
    
    elif reward_type == "human":
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            return 10
        # if slowing down and going higher, reward
        if next_state[0] > state[0][0] and abs(next_state[1]) < abs(state[0][1]):
            return 1
        # if speeding up and going lower, reward
        if next_state[0] < state[0][0] and abs(next_state[1]) > abs(state[0][1]):
            return 1
        else:
            return 0

    elif reward_type=="test": #leave this space for further test mode
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
            reward +=1.5
        
        # give more reward if the cart reaches the flag in 200 steps
        if next_state[0] >= 0.5:
            reward += 1000
        # give more reward if cart is near flag
        elif next_state[0] >= 0.35:
            reward += 100
        else:
            # put a penalty if the no of time steps is more
            reward -= 1 

        # once epsilon gets small, penalize everytime it crosses the valley
        if next_state[0] == -0.5 and agent.epsilon <= agent.epsilon_min:
            reward -= 10

        return reward

    else:
        return 0
