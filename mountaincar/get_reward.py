def get_reward(state, next_state, reward_type):
    if reward_type == "original":
        if next_state[0] >= 0.5:
            print("Car has reached the goal")
            return 10
        if next_state[0] > -0.4:
            return (1+next_state[0])**2
        return 0
    
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

    else:
        return 0