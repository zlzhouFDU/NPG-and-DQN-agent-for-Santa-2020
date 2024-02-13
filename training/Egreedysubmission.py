
import json
import numpy as np
import pandas as pd
import random


import json
import numpy as np
import pandas as pd

bandit_state = None
total_reward = 0
last_step = None

epsilon = 0.17
    
def epsilon_greedy_agent (observation, configuration):
    global history, history_bandit
    
    step = 1.0 #you can regulate exploration / exploitation balacne using this param
    decay_rate = 0.97 # how much do we decay the win count after each call
    
    global bandit_state,total_reward,last_step
        
    if observation.step == 0:
        # initial bandit state
        epsilon = 0.2
        bandit_state = [[1,1] for i in range(configuration["banditCount"])]
    else:       
        epsilon = 0.2
        if observation.step > 100:
            epsilon = 1 / (observation.step)
        # updating bandit_state using the result of the previous step
        last_reward = observation["reward"] - total_reward
        total_reward = observation["reward"]
        
        # we need to understand who we are Player 1 or 2
        player = int(last_step == observation.lastActions[1])
        
        if last_reward > 0:
            bandit_state[observation.lastActions[player]][0] += last_reward * step
        else:
            bandit_state[observation.lastActions[player]][1] += step
        
        bandit_state[observation.lastActions[0]][0] = (bandit_state[observation.lastActions[0]][0] - 1) * decay_rate + 1
        bandit_state[observation.lastActions[1]][0] = (bandit_state[observation.lastActions[1]][0] - 1) * decay_rate + 1

#   generate random answer epsilon% of the time
    if random.random() <= epsilon:
        return random.randrange(configuration.banditCount)
     
#   determine the MLE of Expectation(Success) for each agent and select the highest one
    best_proba = -1
    best_agent = None
    for k in range(configuration["banditCount"]):
        proba = (bandit_state[k][0])/(bandit_state[k][1] + bandit_state[k][0])
        if proba > best_proba:
            best_proba = proba
            best_agent = k
        
    last_step = best_agent
    return best_agent
