import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#User libraries
#from filename import class
from mab_base import mab_base
from greedy_algo import greedy 
from collab_filter import MF

#Python libaries
from math import *



#STEP1: Read csvs and create nice DS out of them
	
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

# Merge the two files
ratings = pd.merge(movies,ratings,on='movieId')

# Creating the sparse matrix
userRatings = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').fillna(0)
userRatings.head()

# Creating predicted rating matrix
R = userRatings.values

#Creating the Collab filtering object
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=1)

training_process = mf.train()
# print()
# print("P x Q:")
# print(mf.full_matrix())
# print()
# print("Global bias:")
# print(mf.b)
# print()
# print("User bias:")
# print(mf.b_u)
# print()
# print("Item bias:")
# print(mf.b_i)
# print("MSE:")
# print(mf.mse())

final_predicted_matrix = pd.DataFrame(mf.full_matrix())

# Splitting the dataset

X_train, X_test = train_test_split(final_predicted_matrix, test_size=0.001)

# Creating Clusters

sse = []
list_k = list(range(4, 20))

for k in list_k:
    km = KMeans(n_clusters=k,n_init = 20, n_jobs=-1)
    km.fit(X_train)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


model_knn = KMeans(n_clusters=15, n_init = 20, random_state = 42, n_jobs=-1)
model_knn.fit(X_train)

# new cluster labels

y_km = model_knn.fit_predict(X_train)
final_clusters = model_knn.cluster_centers_

# Final Clusters
final_clusters = pd.DataFrame(final_clusters)

# Creating a new initial column with cluster name
new_col = {'cluster': [i for i in range(1,len(final_clusters)+1)]}
new_col = pd.DataFrame(new_col)
final_clusters = new_col.join(final_clusters)

# Testing Epsilon_greedy

import Epsilon_Greedy as ep

#create greedy mab object

k = final_clusters.shape[0]
mab_0 = ep.eps_bandit()
mab_0.set_clusters(k, 0.1, 1000, final_clusters)

last_reward = 0
for i in range(1,T):
    # pull arm
    #print('Pull ',i)
    print("Movie Recommended:\t", mab_0.pull_arm(),  end='')
    rec_movie_id = mab_0.pull_arm()
    #mab_0.pull_arm(last_reward)
    #thompsonState(mab_0)
    print('')
    # user fake review
    last_reward = getRewards(X_test,rec_movie_id) 
    print("last_reward: ", last_reward)

# reset
mab_0.reset()
print('')
