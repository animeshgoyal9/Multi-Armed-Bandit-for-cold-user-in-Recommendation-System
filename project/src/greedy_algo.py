from mab_base import mab_base
class greedy(mab_base):
		
	def set_clusters(self,clusters):
		mab_base.clusters = clusters
	def init(self):
		print("From greedy")
	def pull_arm(self):
		print("From greedy")
	def reset(self):
		print("From greedy")

