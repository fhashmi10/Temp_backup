# Import routines
import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # To and From locations in action space
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        # Current location, hour of the day, and day of the week in state space 
        self.state_space = [(i,j,k) for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""
        state_encod = [0 for i in range(m+t+d)]
        
        #Store to variables for clarity
        loc_curr = state[0]
        hour_of_day = state[1]
        day_of_week = state[2]

        # Format it as one hot encoding - value vector to 1 based on indexes
        state_encod[loc_curr] = 1
        state_encod[m+hour_of_day] = 1
        state_encod[m+t+day_of_week] = 1

        return state_encod



    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""

        state_encod = [0 for i in range(m+t+d+m+m)]
        
        #Store to variables for clarity
        loc_curr = state[0]
        hour_of_day = state[1]
        day_of_week = state[2]
        loc_pick = action[0]
        loc_dest = action[1]

        # Format it as one hot encoding - value vector to 1 based on indexes
        state_encod[loc_curr] = 1
        state_encod[m+hour_of_day] = 1
        state_encod[m+t+day_of_week] = 1
        if action!=(0,0):
            state_encod[m+t+d+loc_pick] = 1
            state_encod[m+t+d+m+loc_dest] = 1

        return state_encod


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        #Distribution as per MDP
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)
        
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append((0,0))

        return possible_actions_index,actions   



    # Time matrix function
    def get_times_from_matrix(self, state, action, Time_matrix):
        # Default times to 0
        time_pick_to_dest = 0
        time_curr_to_pick = 0

        if action==(0,0): #Driver not accepting any ride
            time_curr_to_pick = 1 # 1 hour ideal time added
        else:
            # Store required data in variables for clarity
            loc_pick = action[0]
            loc_dest = action[1]
            loc_curr = state[0]
            hour_of_day = state[1]
            day_of_week = state[2]
            # Calculate times
            time_pick_to_dest = int(Time_matrix[loc_pick][loc_dest][hour_of_day][day_of_week])
            time_curr_to_pick = int(Time_matrix[loc_curr][loc_pick][hour_of_day][day_of_week])
           #Time for curr location to pick will be 0 when driver is already at pickup location

        return time_pick_to_dest,time_curr_to_pick

    

    # Reward function
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""

        # Time 0,1 will be returned when no ride is accepted: action(0,0): Reward -C
        # Time x,0 will be returned when driver is already at pick up point
        # Time x,y will be returned when driver is not at pick up point
        time_pick_to_dest,time_curr_to_pick = self.get_times_from_matrix(state, action, Time_matrix)
        reward = (R*time_pick_to_dest)-(C*(time_pick_to_dest+time_curr_to_pick))
    
        return reward


    # hour and day update function
    def get_hour_and_day(self, total_time, hour_day, day_week):
        hour_of_day = hour_day
        day_of_week = day_week

        days_to_add = 0
        hours_to_add = 0

        # Calculate hour and day based on total time
        if total_time>24:
            hours_to_add = int(total_time%24)
            hour_of_day+=hours_to_add
            days_to_add = int(total_time/24)
            day_of_week+=days_to_add
        else:
            hour_of_day+=total_time


        # Modify day and hour if hour is greater than 24 
        if hour_of_day>24:
            days_to_add = int(hour_of_day/24)
            day_of_week+=days_to_add
            # Modify day if day is greater than 7
            if day_of_week>7:
                day_of_week=int(day_of_week%7)
            
            hour_of_day=int(hour_of_day%24)

        return hour_of_day, day_of_week


    
    # Next State function
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        time_pick_to_dest,time_curr_to_pick = self.get_times_from_matrix(state, action, Time_matrix)
        total_time = time_pick_to_dest + time_curr_to_pick

        # Store required data in variables for clarity
        loc_dest = action[1]
        hour_of_day = state[1]
        day_of_week = state[2]
        
        next_state = (loc_dest,self.get_hour_and_day(total_time,hour_of_day,day_of_week))

        return next_state



    #Reset function
    def reset(self):
        return self.action_space, self.state_space, self.state_init
