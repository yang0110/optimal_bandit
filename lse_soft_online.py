import numpy as np 
from scipy.special import softmax
from numpy.random import choice
from sklearn.preprocessing import Normalizer, MinMaxScaler


class LSE_soft_online():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, step_size_beta, weight1, beta, gamma, time_width):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=user_feature
		self.true_payoffs=true_payoffs
		self.item_features=item_features
		self.alpha=alpha
		self.sigma=sigma
		self.weight1=weight1
		self.time_width=time_width
		self.beta=beta
		self.a=beta
		self.gamma=gamma
		self.step_size_beta=step_size_beta
		self.user_f=np.zeros(self.dimension)
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.s_list=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		self.low_bound_list=np.zeros(self.item_num)
		self.soft_max_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.gamma_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.est_y_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_grad=0
		self.gamma_grad=0
		self.lagrange_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.g_s_matrix=np.zeros((self.item_num, self.iteration))
		self.s_matrix=np.zeros((self.item_num, self.iteration))
		self.u_set=[]
		self.l_set=[]


	def update_feature(self,x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def select_arm(self, time, old_payoff):
		self.s_list=np.zeros(self.item_num)
		temp=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		est_y_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		self.low_bound_list=np.zeros(self.item_num)
		for i in range(self.item_num):
			x=self.item_features[i]
			self.x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y_list[i]=np.dot(self.user_f, x)
			self.low_bound_list[i]=np.dot(self.user_f, x)-self.beta*np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.est_y_matrix[time, i]=np.dot(self.user_f, x)-old_payoff

		for j in range(self.item_num):
			index=np.argmax(self.low_bound_list)
			self.x_norm_list2[j]=self.x_norm_list[j]+self.x_norm_list[index]
			self.s_list[j]=self.beta*(self.x_norm_list2[j])-np.abs(est_y_list[index]-est_y_list[j])
			self.s_matrix[j, time]=self.s_list[j]
			self.g_s_matrix[j,time]=self.gamma*self.s_list[j]
			if self.s_list[j]>=0:
				self.u_set.extend([j])
			else:
				self.l_set.extend([j])


		soft_max=np.exp(self.gamma*self.s_list)/np.sum(np.exp(self.gamma*self.s_list))
		self.soft_max_matrix[time]=soft_max
		ind=choice(range(self.item_num), p=soft_max)
		x=self.item_features[ind]
		payoff=self.true_payoffs[ind]
		# +np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[ind]
		return ind, x, payoff, regret

	def find_lagrange_grad(self, time):
		for i in range(self.item_num):
			self.lagrange_grad_matrix[time, i]=self.weight1*self.x_norm_list[i]


	def find_log_grad_beta(self, time):
		a=self.gamma*np.dot(self.x_norm_list2, np.exp(self.gamma*self.s_list))
		b=np.sum(np.exp(self.gamma*self.s_list))
		for i in range(self.item_num):
			self.beta_log_grad_matrix[time, i]=self.gamma*(self.x_norm_list2[i])-a/b


	def update_gamma(self, time):
		cl=len(np.unique(self.l_set))
		delta=0.99
		if cl==0:
			cl=1
		else:
			pass
		a=np.log((delta*cl)/(1-delta))
		max_s=np.max(self.s_list)
		self.gamma=a/max_s
		if time<=self.time_width:
			self.gamma=0
		else:
			pass

	def update_beta_grad(self, time):
		self.beta_grad=0
		for t in range(time-self.time_width, time):
			if t<0:
				pass 
			else:
				temp_2=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_matrix[t], self.soft_max_matrix[t], self.beta_log_grad_matrix[t], self.lagrange_grad_matrix[t])])
				self.beta_grad+=temp_2

	def update_beta(self, time):
		if (1/(time+1))>self.step_size_beta:
			pass
		else:
			self.step_size_beta=1/(2*time+1)
		self.beta+=self.step_size_beta*self.beta_grad
		if time<=self.time_width:
			self.beta=self.a
		else:
			pass

	def run(self):
		error_list=np.zeros(self.iteration)
		cum_regret=[0]
		old_payoff=0
		beta_list=np.zeros(self.iteration)
		old_payoff=0
		for time in range(self.iteration):
			print('time/iteration, %s/%s beta=%s ~~~~~ LSE-Soft-Online'%(time, self.iteration, np.round(self.beta, decimals=2)))
			ind, x, y, regret=self.select_arm(time, old_payoff)
			old_payoff=y
			self.update_feature(x, y)
			self.update_gamma(time)
			self.find_log_grad_beta(time)
			self.find_lagrange_grad(time)
			self.update_beta_grad(time)
			self.update_beta(time)
			beta_list[time]=self.beta
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=np.linalg.norm(self.user_f-self.user_feature)

		return cum_regret[1:], error_list, self.soft_max_matrix.T, self.s_matrix, beta_list

		














