# Inspired by:
# Sourcing from: http://www.aionlinecourse.com/tutorial/machine-learning/upper-confidence-bound-%28ucb%29
# https://www.datahubbs.com/multi-armed-bandits-reinforcement-learning-2/ 
import numpy as np
from mab_base import mab_base


class ucb_bandit(mab_base):
    '''
    Upper Confidence Bound k-bandit problem
    
    Inputs 
    ============================================
    k: number of arms (int)
    c:
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    def __init__(self):
        # Step count
        self.n = 0
        # Previous arm pulled
        self.last_pick = 0

    def set_clusters(self,clusters):
        # Number of arms k=d, N=n
        self.k = clusters.shape[0]
        # Step count for each arm (n_i(n))
        self.k_n = np.zeros(clusters.shape[0])
        # Mean reward for each arm
        self.reward = np.zeros(clusters.shape[0])
        # Reward for each arm (r_i(n))
        self.k_reward = np.zeros(clusters.shape[0])
        # Delta for each arm (r_i(n))
        self.k_delta = np.zeros(clusters.shape[0])
        
    def pull_arm(self, last_reward=None):
        # Update rewards for last pick
        # Update counts
        self.n += 1
        self.k_n[self.last_pick] += 1 #increment n_i since selected
        print(self.last_pick)
        self.k_reward[self.last_pick] += last_reward
        self.reward = self.k_reward[self.last_pick]/self.k_n[self.last_pick] #r_i
        # Calculate delta:
        self.k_delta[self.last_pick] = np.sqrt((3/2) * (np.log(self.n + 1) / self.k_n[self.last_pick]))
       
        # n_arms = len(self.k)
        for arm in range(self.k):
            if self.k_n[arm] == 0:
                self.last_pick = arm
                # print(arm)
                return arm
            
        # Select action according to UCB Criteria
        a = np.argmax(self.reward + self.k_delta)
        # print(a)
        # This is last_pick for next iteration, required to update reward
        self.last_pick = a   
        return a
        
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.reward = np.zeros(self.k)
        self.k_reward = np.zeros(self.k)
        self.k_delta = np.zeros(self.k)

# sudo-code: constant: d arms N rounds
# global:
# n_i(n)- #of times add i was selected up to round n
# r_i(n)- sum of rewards of add i up to n

# avg_r_i(n) - r_i(n)/n_i(n)
# conf_i(n) - avg_r_i(n)-c(n)*avg_r_i(n)+c(n)
# c=sqrt((3*log(n))/(2*n_i(n)))

# select->argmax(avg_r_i(n)+c(n)) - for that round