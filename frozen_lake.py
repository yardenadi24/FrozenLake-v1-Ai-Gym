import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake8x8-v1")

gamma = 0.95
num_actions = 4
num_states = 64
def eps_greedy_policy(Q,epsilon,s):
    action_prob = np.ones(num_actions,dtype=float) * epsilon /num_actions
    greedy_action = np.argmax(Q[s])
    action_prob[greedy_action] +=(1.0-epsilon)
    return action_prob

