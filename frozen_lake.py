
import gym
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys  
from collections import defaultdict
import plotting
import copy
  
matplotlib.style.use('ggplot')
env = gym.make("FrozenLake8x8-v1")

gamma = 0.95


#epsilon greedy policy that depends on a specific Q and epsilon
#returns a lambda : (state-> list of actions and their prob)
def createEpsGreedyPolicy(Q, epsilon, num_actions):

    def getPolicy(state):
        
        #probabilty for ech action form state s
        policy_prob_s = np.ones(
            num_actions, dtype = float) * epsilon / num_actions
                  
        best_action = np.argmax(Q[state])

        #best action must have better probability
        policy_prob_s[best_action] += (1.0 - epsilon)
        return policy_prob_s
   
    return getPolicy

def qLearningAlgo(env, num_episodes, gamma ,alpha , epsilon, lamda):
     
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
   
       
    # Create an epsilon greedy policy function
    policy = createEpsGreedyPolicy(Q, epsilon, env.action_space.n)
    iter = 1
    improve_stats={}


      
    # For every episode
    for ith_episode in range(num_episodes):
           
        state = env.reset()

        #set new eligibility traces for this new episode
        eligibility_traces = defaultdict(lambda: np.zeros(env.action_space.n))

        #itterate infinitly or until reaching 260 steps   
        for t in itertools.count():
            
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to the probability distribution
            action = np.random.choice([0,1,2,3],p = action_probabilities)
   
            #we got the state and the action now updated eligibility traces
            eligibility_traces[state][action] += 1
            eligibility_traces_factor = eligibility_traces[state][action]   
            # take action
            next_state, reward, done, _ = env.step(action)
               
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * eligibility_traces_factor*td_delta

            #update eligibility traces for state and action taken
            eligibility_traces[state][action] = lamda * eligibility_traces[state][action]

            # calculate policy
            if(iter%10000 == 0):
                sim_env = copy.deepcopy(env)
                #TO DO- policy value is the mean of "n" simulations
                policy_val = eval_policy(env,Q,policy)
                improve_stats[iter] = policy_val
            # done is True if episode terminated   
            if done:
                break

            iter +=1       
            state = next_state
       
    return Q


#evaluate a policy value
def eval_policy(env_1,Q,policy):
    state = env_1.reset()
    discounter_reward = 0
    for t in itertools.count():
        best_action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(best_action)
        discounter_reward += reward
#TO DO - save stats for simulation print -
        if done:
            break
        if (t>300):
            break
#TO DO- return stats to print and reward           
    return 0

def main():
    return 0
   #TO DO: 4 permutation

if __name__ == "__main__":
    main()    