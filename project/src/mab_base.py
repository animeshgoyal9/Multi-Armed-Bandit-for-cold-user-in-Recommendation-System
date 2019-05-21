import pandas as pd
class mab_base:
	clusters = pd.DataFrame()
	def set_clusters(self,clusters):
		self.clusters = clusters
	def init(self): #initialize all variables for your class
		pass
	def pull_arm(self, last_reward):#pull the arm, update the table
		pass
	def reset(self):
		pass





##Global
def get_reward():#@Animesh-Implement the reward functionality here while implementing the framework
	return(0)



