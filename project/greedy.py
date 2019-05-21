import numpy as np
import random
from random import choices
cluster =[0,1,2]
best_arm = 0
total_reward = 0
rewards = [] #holds rewards for every arm
count = []  #holds count of arm selected in every draw(to find average)
avg_rewards=[]
def init():
	best_arm = random.randrange(len(cluster))    #initalize best arm randomly!
	print("Initializing best arm to ",best_arm )
	assert(best_arm < len(cluster))
	total_reward = 0	
	for i in range(0,len(cluster)):
		rewards.append(0)
		count.append(0)


def get_rewards(arm): ##TODO To be replaced with a function which calculates the rewards for the chosen arm?
	#for the chosen arm, compare with the true data and calculate reward 	#for now, returning a random reward
	return(np.random.randint(3000))

def exploit():  #exploit will simply choose the best arm and 'exploit' it ie. get rewards from this arm
	return(get_rewards(best_arm))

def explore():#will simply randomly choose another arm and get rewards for that
	arm = (random.randrange(0,len(cluster)))
	rewards[arm] = get_rewards(arm)
	count[arm] = count[arm] + 1
	avg_rewards=[]
	for i in range(0,len(rewards)):
		if(count[i] == 0):
			print("0")
			avg_rewards.append(0)	
		else:
			print(rewards[i]/count[i])
			avg_rewards.append(rewards[i]/count[i])
	
	max_value = max(avg_rewards)
	assert(len(avg_rewards) == len(cluster))
	best_arm = avg_rewards.index(max_value)
	print("New best arm to ",best_arm )
	assert(best_arm < len(cluster))
	#return the reward as well for this round to add to the final count
	return(rewards[arm])

def choose_action():
	global total_reward
	population = [0,1]   # 0 - explore, 1 - exploit
	weights = [0.5, 0.5] #epsilon value
	for i in range(1,100):
		switch = choices(population,weights)
		if(switch[0] == 0):
			print("**************Explore**************")
			temp = explore()
			total_reward = total_reward +  temp
		else:
			print("**************Exploit***************")
			temp = exploit()
			total_reward = total_reward +  temp
#	print(total_reward)





init()
choose_action()
