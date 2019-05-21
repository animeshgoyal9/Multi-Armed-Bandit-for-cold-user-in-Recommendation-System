import numpy as np
import pandas as pd

import thompson_algo as ta

def printClusters(mab):
    print('Current Cluster State:')
    print(mab.clusters)
    print('')
    print('Original/Safe Clusters:')
    print(mab.clusters_safe)

def thompsonState(mab):
    print('--- MAB Status -------------------------------------------')
    print('Arms:\t\t', end = '')
    print(mab.k)
    print('')
    print('Iterations:\t', end='')
    print(mab.n)
    print('')
    print('Arm Alpha:\t', end='')
    print(mab.k_a)
    print('')
    print('Arm Beta:\t', end='')
    print(mab.k_b)
    print('')
    print('Arm Theta:\t', end='')
    print(mab.k_theta)
    print('')
    print('Arm Reward:\t', end='')
    print(mab.k_reward)
    print('')
    printClusters(mab)
    print('----------------------------------------------------------')
    

# make clusters
cluster_list = [['theta_1', 4.40, 2.94, 4.32, 4.62, 4.85, 1.67, 1.10],
                ['theta_2', 1.64, 3.74, 2.73, 4.16, 1.91, 1.83, 1.39]]

clusters = pd.DataFrame(cluster_list, columns = ['cluster', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7'])

print(clusters)
print('')

# create thompson mab object
mab_0 = ta.thompson()
mab_0.set_clusters(clusters)
thompsonState(mab_0)

# pull arm
print('Pull 1')
print("Movie Recommended:\t", end='')
print(mab_0.pull_arm())
thompsonState(mab_0)
print('')

# user fake review
last_reward = 3

# pull arm
print('Pull 2')
print("Movie Recommended:\t", end='')
print(mab_0.pull_arm(last_reward))
thompsonState(mab_0)
print('')

# user fake review
last_reward = 4.5

# pull arm
print('Pull 3')
print("Movie Recommended:\t", end='')
print(mab_0.pull_arm(last_reward))
thompsonState(mab_0)
print('')

# reset
mab_0.reset()
thompsonState(mab_0)
print('')






