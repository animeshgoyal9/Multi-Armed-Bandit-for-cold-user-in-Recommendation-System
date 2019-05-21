import numpy as np
import pandas as pd
import math


#PARAMETERS
T=5  #Number of iterations for a given run
N = 2 #Number of runs
DEBUG_PRINT = 1

#Use this to implement the "framework"


#User libraries
#from filename import class
from mab_base import mab_base
from collab_filter import MF
import thompson_algo as ta
import UCB as ucb
import Epsilon_Greedy as ep


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



dcg_thom=[]
dcg_ucb = []
dcg_greedy = []
dcg_thom_true=[]
dcg_ucb_true = []
dcg_greedy_true = []

def DCG_Normalize(dcg_true,dcg_pred):
    assert(len(dcg_true) == len(dcg_pred) )
    for index in range(len(dcg_true)) :
        ndcg += dcg_pred[index]/dcg_user[index]
    return ndcg/len(dcg_true)

for i in range(1,N):
    #Add cluster classification here
    # Creating Clusters

    #LOO here 
    X_train, X_test = train_test_split(final_predicted_matrix, test_size=0.001)
    print(X_test)


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


    model_knn = KMeans(n_clusters=3, n_init = 20, random_state = 42, n_jobs=-1)
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
    print("Done")

    # create thompson mab object
    r_u_t_thom  = []
    r_u_t_thom_true  = []

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
        #thompsonState(mab_0)
        print('')
        last_reward = getRewards(X_test,rec_movie_id) 
        #print("last_reward: ", last_reward)
        temp = X_test.iloc[:,rec_movie_id:rec_movie_id+1]
        #print(temp[rec_movie_id].values[0])
        r_u_t_thom.append(temp[rec_movie_id].values[0])
        r_u_t_thom_true.append(np.max(X_test)[0])


    dcg_thom.append(DCG_Pred_perUser(r_u_t_thom))
    dcg_thom_true.append(DCG_Pred_perUser(r_u_t_thom_true))
    print("True DCG for current user for Thompson: ", dcg_thom_true)
    print("DCG for current user for Thompson: ", dcg_thom)


    # reset
    mab_0.reset()
    #thompsonState(mab_0)
    print('')

    # create ucb mab object
    r_u_t_ucb = []
    r_u_t_ucb_true = []
    mab_0 = ucb.ucb_bandit()
    mab_0.set_clusters(final_clusters)
    #thompsonState(mab_0)
    last_reward = 0
    for i in range(1,T):
        # pull arm
        #print('Pull ',i)
        print("Movie Recommended:\t", end='')
        rec_movie_id = mab_0.pull_arm(last_reward)
        #mab_0.pull_arm(last_reward)
        print('')
        last_reward = getRewards(X_test,rec_movie_id) 
        #print("last_reward: ", last_reward)
        temp = X_test.iloc[:,rec_movie_id:rec_movie_id+1]
        #print(temp[rec_movie_id].values[0])
        r_u_t_ucb.append(temp[rec_movie_id].values[0])
        r_u_t_ucb_true.append(np.max(X_test)[0])



    dcg_ucb.append(DCG_Pred_perUser(r_u_t_ucb))
    dcg_ucb_true.append(DCG_Pred_perUser(r_u_t_ucb_true))
    print("True DCG for current user for UCB: ", dcg_ucb_true)
    print("DCG for current user for UCB: ", dcg_ucb)

    # reset

    mab_0.reset()
    print('')

    #Greedy object
    r_u_t_greedy = []
    r_u_t_greedy_true = []
    #   Implementing equation 4
    def DCG_Pred_perUser(r_u_t):
        dcg = r_u_t[0]
        for time in range(len(r_u_t)) :
            if (time > 1) :
                #print(time)
                dcg += r_u_t[time]/(math.log(time,2))
        return dcg


    k = final_clusters.shape[0]
    mab_0 = ep.eps_bandit()
    mab_0.set_clusters(final_clusters)
    last_reward = 0
    for i in range(1,T):
        # pull arm
        #print('Pull ',i)
        print("Movie Recommended:\t",end='')
        rec_movie_id = mab_0.pull_arm(last_reward)
        #mab_0.pull_arm(last_reward)
        #thompsonState(mab_0)
        print('')
        last_reward = getRewards(X_test,rec_movie_id)
        #print("last_reward: ", last_reward)
        temp = X_test.iloc[:,rec_movie_id:rec_movie_id+1]
        #print(temp[rec_movie_id].values[0])
        r_u_t_greedy.append(temp[rec_movie_id].values[0])
        r_u_t_greedy_true.append(np.max(X_test)[0])


    dcg_greedy.append(DCG_Pred_perUser(r_u_t_greedy))
    dcg_greedy_true.append(DCG_Pred_perUser(r_u_t_greedy_true))
    print("True DCG for current user for Greedy: ", dcg_greedy_true)
    print("DCG for current user for Greedy: ", dcg_greedy)

    # reset
    mab_0.reset()
    print('')



def DCG_Normalize(dcg_pred,dcg_true):
    assert(len(dcg_true) == len(dcg_pred) )
    ndcg = 0
    for index in range(len(dcg_true)) :
        ndcg += dcg_pred[index]/dcg_true[index]
    return ndcg/len(dcg_true)

ndg_greedy = DCG_Normalize(dcg_greedy,dcg_greedy_true)
ndg_ucb = DCG_Normalize(dcg_ucb,dcg_ucb_true)
ndg_thom = DCG_Normalize(dcg_thom,dcg_thom_true)

print("NDCG for Greedy Algorithm for ",N, "users over time ",T, ": ", ndg_greedy)
print("NDCG for UCB Algorithm for ",N, "users over time ",T, ": ", ndg_ucb)
print("NDCG for Thompson Algorithm for ",N, "users over time ",T, ": ", ndg_thom)

