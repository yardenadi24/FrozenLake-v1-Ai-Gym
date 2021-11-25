
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
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):

    def policyFunction(state):
   
        policy_prob_s = np.ones(
            num_actions, dtype = float) * epsilon / num_actions
                  
        best_action = np.argmax(Q[state])
        policy_prob_s[best_action] += (1.0 - epsilon)
        return policy_prob_s
   
    return policyFunction

def qLearningAlgo(env, num_episodes, gamma ,
                            alpha , epsilon):
    """
    Q-Learning algorithm: Off-policy TD control.

    """
       
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
   
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))    
       
    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
    iter = 1
    improve_stats={}   
    # For every episode
    for ith_episode in range(num_episodes):
           
        # Reset the environment and pick the first action
        state = env.reset()

        #itterate infinitly   
        for t in itertools.count():
               
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
   
            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)
   
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t
               
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
   

            # calculate policy
            if(iter%10000 == 0):
                sim_env = copy.deepcopy(env)
                #TO DO- policy value is the mean of "n" simulations
                policy_val = eval_policy(env,Q,policy)
                improve_stats[iter] = policy_val
            # done is True if episode terminated   
            if done:
                break
                   
            state = next_state
       
    return Q, stats

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
    Q_eps_095_apha_1,stats_1 = qLearningAlgo(env,250,gamma,1,0.995)
    Q_eps_095_apha_2,stats_1 = qLearningAlgo(env,250,gamma,1,0.995)

if __name__ == "__main__":
    main()    