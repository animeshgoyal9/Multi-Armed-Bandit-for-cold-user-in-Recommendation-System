import numpy as np
from mab_base import mab_base
'''
Inspired by:

https://www.datahubbs.com/multi_armed_bandits_reinforcement_learning_1/
'''
class eps_bandit(mab_base):
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    '''

    def __init__(self):
        # Step count
        self.n = 0
        # Total mean reward
        self.mean_reward = 0
        # Previous arm pulled
        self.last_pick = -1
        
    def set_clusters(self, clusters):
        # Number of arms
        self.k = clusters.shape[0]
        # Search probability
        self.eps = 0.1
        # Step count for each arm
        self.k_n = np.zeros(self.k)
        # Mean reward for each arm
        self.k_reward = np.zeros(self.k)
        # Clusters
        self.clusters = clusters
        self.clusters_safe = clusters.copy()
            
    def pull_arm(self,last_reward= None):
        # Generate random number
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)

        # Update counts
        self.n += 1
        self.k_n[a] += 1
    
        # Update total
        self.mean_reward = self.mean_reward + (last_reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (last_reward - self.k_reward[a]) / self.k_n[a]
            
        temp = self.clusters.iloc[a]
        temp = np.array(temp.drop('cluster'))
        rec_idx = np.argmax(temp) + 1 # +1 because the first column is for cluster name
        #movieIds = self.clusters.columns
        #rec = movieIds[rec_idx]
        rec = self.clusters.columns[rec_idx]
        #rec = self.clusters.iloc[p, rec_idx]
        #self.clusters.iloc[:, rec_idx] = 0
        
        return rec
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.k_reward = np.zeros(self.k)
        self.clusters = self.clusters_safe.copy()
