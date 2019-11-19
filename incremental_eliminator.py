import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class INC_ELI():
	def __init__(self, dimension, iteration, true_user_feature, item_num, item_feature_matrix, true_payoffs, alpha, sigma, delta):
		self.phase_num=np.int(np.ceil(np.log2(iteration)))
		self.iteration=iteration
		self.dimension=dimension
		self.item_num=item_num
		self.true_user_feature=true_user_feature
		self.item_feature_matrix=item_feature_matrix
		self.item_list=list(range(self.item_num))
		self.true_payoffs=true_payoffs
		self.user_feature=np.zeros(self.dimension)
		self.alpha=alpha
		self.sigma=sigma
		self.delta=delta
		self.cov=np.zeros((self.dimension, self.dimension))
		self.bias=np.zeros(self.dimension)
		self.beta=0 
		self.cum_regret=[0]
		self.item_index=[]
		self.remained_item=[]
		self.beta_list=[]
		self.error=[]
		self.x_norm_matrix=np.zeros((self.iteration, self.item_num))
		self.index_matrix=np.zeros((self.iteration, self.item_num))
		self.mean_matrix=np.zeros((self.iteration, self.item_num))
		self.ucb_matrix=np.zeros((self.iteration, self.item_num))
		self.payoff_error_matrix=np.zeros((self.iteration, self.item_num))
		self.ucb_list=np.zeros(self.iteration)
		self.true_ucb_list=np.zeros(self.iteration)

	def update_beta(self, time):
		self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(time/self.delta)))+np.sqrt(self.alpha)*np.linalg.norm(self.true_user_feature)
	def update_time_in_phase(self, phase):
		start_time=2**(phase-1)
		end_time=2**(phase)-1
		if end_time>=self.iteration:
			end_time=self.iteration
		else:
			pass
		return start_time, end_time
	def select_arm(self, phase, start_time, end_time):
			x_norm_list=np.zeros(self.item_num)
			for t in range(start_time, end_time):
				self.update_beta(t)
				self.beta_list.extend([self.beta])
				print('phase %s, time %s, iteration %s, item_num %s'%(phase, t, self.iteration, len(self.item_list)), '~~~ INCRE ELIMINATOR')
				for i in self.item_list:
					x=self.item_feature_matrix[i].copy()
					x_norm=np.sqrt(np.dot(np.dot(x, np.linalg.pinv(self.cov)),x))
					x_norm_list[i]=x_norm
					mean=np.dot(self.user_feature, x)
					self.x_norm_matrix[t, i]=x_norm
					self.ucb_matrix[t,i]=self.beta*x_norm
					self.index_matrix[t,i]=mean+self.beta*x_norm
					self.mean_matrix[t,i]=mean
					self.payoff_error_matrix[t,i]=np.abs(mean-self.true_payoffs[i])


				max_index=np.argmax(x_norm_list)
				x_t=self.item_feature_matrix[max_index]
				x_t_norm=np.sqrt(np.dot(np.dot(x_t, np.linalg.pinv(self.cov)),x_t))
				self.ucb_list[t]=self.beta*x_t_norm
				self.true_ucb_list[t]=np.abs(np.dot(self.user_feature, x_t)-self.true_payoffs[max_index])

				self.item_index.extend([max_index])
				payoff=self.true_payoffs[max_index]
				max_payoff=np.max(self.true_payoffs)
				regret=max_payoff-payoff 
				self.cum_regret.extend([self.cum_regret[-1]+regret])
				self.cov+=np.outer(x_t,x_t)
				self.bias+=x_t*payoff
				self.user_feature=np.dot(np.linalg.pinv(self.cov), self.bias)
				self.error.extend([np.linalg.norm(self.user_feature-self.true_user_feature)])

	def update_model(self):
		self.user_feature=np.dot(np.linalg.pinv(self.cov), self.bias)

	def eliminate_arms(self, phase):
		index_list=np.zeros(self.item_num)
		index_list2=np.zeros(self.item_num)
		for i in self.item_list:
			x=self.item_feature_matrix[i]
			x_norm=np.sqrt(np.dot(np.dot(x, np.linalg.pinv(self.cov)),x))
			index_list[i]=np.dot(self.user_feature, x)+self.beta*x_norm 
			self.index_matrix[phase-1, i]=np.dot(self.user_feature, x)+self.beta*x_norm 
			index_list2[i]=np.dot(self.user_feature, x)-self.beta*x_norm 
		p=np.max(index_list2)
		self.item_list=list(np.where(np.array(index_list)>=p)[0])
		print('remained item num ~~~~', len(self.item_list))
		self.remained_item.extend([len(self.item_list)])

	def run(self):
		for j in range(self.phase_num):
			phase=j+1
			start_time, end_time=self.update_time_in_phase(phase)
			self.cov=self.alpha*np.identity(self.dimension)
			self.bias=np.zeros(self.dimension)
			self.select_arm(phase, start_time, end_time)
			self.update_model()
			self.eliminate_arms(phase)

		return self.cum_regret, self.error, self.item_index, self.remained_item, self.beta_list, self.x_norm_matrix, self.index_matrix, self.ucb_matrix, self.mean_matrix, self.payoff_error_matrix, self.ucb_list, self.true_ucb_list

