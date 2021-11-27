
import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import copy
  
env = gym.make("FrozenLake8x8-v1")
gamma = 0.95
actions = ['move_left','move_down','move_right','move_up']


# epsilon greedy policy that depends on a specific Q and epsilon
# returns a lambda : (state-> list of actions and their prob)
def createEpsGreedyPolicy(Q, epsilon, num_actions):
    def getPolicy(state):
        # probability for each action form state s
        policy_prob_s = np.ones(
            num_actions, dtype = float) * epsilon / num_actions
                  
        best_action = np.argmax(Q[state])

        # best action must have better probability
        policy_prob_s[best_action] += (1.0 - epsilon)
        return policy_prob_s
   
    return getPolicy


def qLearningAlgo(env, num_episodes, gamma, alpha, epsilon, lamda, n, et):
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Create an epsilon greedy policy function
    policy = createEpsGreedyPolicy(Q, epsilon, env.action_space.n)
    iter = 1
    improve_stats = [[], []]

    # For every episode
    for ith_episode in range(num_episodes):
        # set new eligibility traces for this new episode
        eligibility_traces = defaultdict(lambda: np.zeros(env.action_space.n))

        state = env.reset()
        # iterate infinitely or until reaching 250 steps
        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to the probability distribution
            action = np.random.choice([0,1,2,3],p=action_probabilities)

            # we got the state and the action now updated eligibility traces
            if et:
                eligibility_traces[state][action] += 1
            eligibility_traces_factor = eligibility_traces[state][action]
            # take action
            next_state, reward, done, _ = env.step(action)

            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            if et:
                Q[state][action] += alpha * eligibility_traces_factor*td_delta
            else:
                Q[state][action] += alpha * td_delta

            policy = createEpsGreedyPolicy(Q, epsilon, env.action_space.n)

            # update eligibility traces for state and action taken
            eligibility_traces[state][action] = lamda * gamma * eligibility_traces[state][action]

            # calculate policy
            if iter % 10000 == 0:
                sim_env = copy.deepcopy(env)
                # policy value is the mean of "n" simulations
                policy_val = eval_policy(sim_env, Q, n)
                improve_stats[0].append(iter)
                improve_stats[1].append(policy_val)
                if policy_val>0:
                    epsilon = epsilon * 0.9
            # done is True if episode terminated   
            if done or t>350:
                iter += 1
                break

            iter += 1       
            state = next_state

        if ith_episode > num_episodes or iter > 1000000:
            break

    return Q, improve_stats


# evaluate a policy value
def eval_policy(env_1,Q,n):
    sum = 0
    for iter in range(0,n):
        part_sum = simulate(env_1,Q)
        sum += part_sum

    return sum/n


def simulate(env_1, Q):
    state = env_1.reset()
    discounter_reward = 0
    for _ in itertools.count():
        best_action = np.argmax(Q[state])
        next_state, reward, done, _ = env_1.step(best_action)
        discounter_reward += reward

        state = next_state
        if done:
            break

    return discounter_reward


def print_simulate(env_1, Q):
    state = env_1.reset()
    total_reward = 0
    steps = 0
    for i in itertools.count():
        steps += 1
        best_action = np.argmax(Q[state])
        actions.append(best_action)
        next_state, reward, done, _ = env_1.step(best_action)
        total_reward += reward
        row = int(state/8)
        col = state % 8
        sign = '+' if reward > 0 else ''
        print(f'{i+1}. {row},{col},{env.desc[row, col]} 7,7 {actions[best_action]} {sign}{reward}')
        state = next_state
        if done:
            break
    print(f'total steps: {steps}')
    total_reward_sign = '+' if total_reward > 0 else ''
    print(f'total rewards: {total_reward_sign}{total_reward}')


def make_Graph(stats1,stats2,stats3,stats4):
    plt.plot(stats1[0],stats1[1], label="alpha: 0.1, lambda: 0.1")
    plt.plot(stats2[0],stats2[1], label="alpha: 0.1, lambda: 0.2")
    plt.plot(stats3[0],stats3[1], label="alpha: 0.2, lambda: 0.1")
    plt.plot(stats4[0],stats4[1], label="alpha: 0.2, lambda: 0.2")

    plt.title("value per step")
    plt.ylabel("value")
    plt.xlabel("step")
    plt.legend()
    plt.show()


def Compare_EG(stats1,stats2):
    plt.plot(stats1[0],stats1[1], label="With eligibility traces")
    plt.plot(stats2[0],stats2[1], label="Without eligibility traces")

    plt.title("value per step")
    plt.ylabel("value")
    plt.xlabel("step")
    plt.legend()
    plt.show()


def main():
    Q1, stats1 = qLearningAlgo(env, 100000, gamma, 0.1, 0.995, 0.1, 250, True)
    Q2, stats2 = qLearningAlgo(env, 100000, gamma, 0.1, 0.995, 0.2, 250, True)
    Q3, stats3 = qLearningAlgo(env, 100000, gamma, 0.2, 0.995, 0.1, 250, True)
    Q4, stats4 = qLearningAlgo(env, 100000, gamma, 0.2, 0.995, 0.2, 250, True)
    make_Graph(stats1, stats2, stats3, stats4)
    Q5, stats5 = qLearningAlgo(env, 100000, gamma, 0.1, 0.995, 0.2, 250, False)
    Compare_EG(stats2, stats5)
    print_simulate(env, Q2)
    return 0


if __name__ == "__main__":
    main()
