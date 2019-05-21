import numpy as np

from mab_base import mab_base

'''
inspired by:

https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50
'''


class thompson(mab_base):
    '''
    thompson algorithm k-bandit
    
    Inputs
    =====================================================
    clusters: A pandas dataframe with a cluster for each row.
              The columns represent the avg movie rating vector
              for said cluster. 

    '''
    def __init__(self):
        # Step count
        self.n = 0
        # Total mean reward
        self.mean_reward = 0
        # Previous arm pulled
        self.last_pick = -1

    def set_clusters(self,clusters):
        # Number of arms
        self.k = clusters.shape[0]
        # Step count for each arm
        self.k_n = np.zeros(clusters.shape[0])
        # Alpha for each arm
        self.k_a = np.ones(clusters.shape[0])
        # Beta for each arm
        self.k_b = np.ones(clusters.shape[0])
        # Theta for each arm
        self.k_theta = np.zeros(clusters.shape[0])
        # Mean reward for each arm
        self.k_reward = np.zeros(clusters.shape[0])
        # Clusters
        mab_base.clusters = clusters
        self.clusters_safe = clusters.copy()

    def pull_arm(self, last_reward=None):
        # update rewards and state
        lp = self.last_pick
        if (self.n > 0):
            self.k_a[lp] += last_reward
            self.k_b[lp] += 5 - last_reward

            self.k_reward[lp] = self.k_reward[lp] + (last_reward - self.k_reward[lp]) / self.k_n[lp] 
            self.mean_reward = self.mean_reward + (last_reward - self.mean_reward) / self.n 

        # cluster pick
        self.k_theta = np.random.beta(self.k_a, self.k_b)
        p = np.argmax(self.k_theta)

        # update counts and clusters
        self.n += 1
        self.k_n[[p]] += 1
        self.last_pick = p
        temp = self.clusters.iloc[p]
        temp = np.array(temp.drop('cluster'))
        rec_idx = np.argmax(temp) + 1 # +1 because the first column is for cluster name
        #movieIds = self.clusters.columns
        #rec = movieIds[rec_idx]
        rec = self.clusters.columns[rec_idx]
        #rec = self.clusters.iloc[p, rec_idx]
        #self.clusters.iloc[:, rec_idx] = 0
        return rec
    # TODO need to return movieId, not movieRating

    def reset(self):
        self.n = 0
        self.k_n = np.zeros(self.clusters.shape[0])
        self.mean_reward = 0
        self.k_reward = np.zeros(self.clusters.shape[0])
        self.clusters = self.clusters_safe.copy()



