import numpy as np
import pandas as pd


#PARAMETERS
T=5  #Number of iterations for a given run
N = 2 #Number of runs
DEBUG_PRINT = 1

#Use this to implement the "framework"


#User libraries
#from filename import class
from mab_base import mab_base
from greedy_algo import greedy 
from collab_filter import MF
import thompson_algo as ta

#Python libaries
from math import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#def DCG_compute():
#   
##   Implementing equation 3    
#def DCG_Normalize(self,n_users,dcg_true,dcg_pred):
#	for index in range(len(dcg_true)) :
#		ndcg += dcg_pred[index]/dcg_user[index]
#	return ndcg/n_users
#    
##   Implementing equation 4
#def DCG_Pred_perUser(self,r_u_t):
#	dcg = r_u_t[0]
#	for time in range(len(r_u_t)) :
#		if (time > 0) :
#			dcg += r_u_t[time]/(math.log(time,2)) 

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
    
def getRewards(X_test, rec_movie):  #get true value of current user and the recommended movie/or recommended cluster
    temp = X_test.iloc[:,rec_movie:rec_movie+1]
    reward = temp[rec_movie].values[0]/np.max(X_test)[0]    
    return reward

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
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Global bias:")
print(mf.b)
print()
print("User bias:")
print(mf.b_u)
print()
print("Item bias:")
print(mf.b_i)
print("MSE:")
print(mf.mse())

final_predicted_matrix = pd.DataFrame(mf.full_matrix())
print(final_predicted_matrix)


#For loop
#for i in range(1,N):
#LOO here 
X_train, X_test = train_test_split(final_predicted_matrix, test_size=0.001)
print(X_test)



#Add cluster classification here
# Creating Clusters

sse = []
list_k = list(range(4, 20))

for k in list_k:
    km = KMeans(n_clusters=k,n_init = 20, n_jobs=-1)
    km.fit(X_train)
    sse.append(km.inertia_)

# Plot sse against k
#plt.figure(figsize=(6, 6))
#plt.plot(list_k, sse, '-o')
#plt.xlabel(r'Number of clusters *k*')
#plt.ylabel('Sum of squared distance');


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
#Setting cluster for other algorithms to use
final_clusters = new_col.join(final_clusters)



base = mab_base()

#Testing MAB algo
#mab1 = greedy()
#mab1.init()
#mab1.set_clusters(clusters)

# create thompson mab object
mab_0 = ta.thompson()
mab_0.set_clusters(final_clusters)
#thompsonState(mab_0)
last_reward = 0
for i in range(1,T):
    # pull arm
    #print('Pull ',i)
    print("Movie Recommended:\t", end='')
    rec_movie_id = mab_0.pull_arm(last_reward)
    #mab_0.pull_arm(last_reward)
    thompsonState(mab_0)
    print('')
    # user fake review
    last_reward = getRewards(X_test,rec_movie_id) 
    print("last_reward: ", last_reward)

#Add NDCG calculation here!
    
# reset
mab_0.reset()
thompsonState(mab_0)
print('')



