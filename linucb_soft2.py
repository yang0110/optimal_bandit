import numpy as np 
from scipy.special import softmax
from numpy.random import choice
from sklearn.preprocessing import Normalizer, MinMaxScaler

class LinUCB_soft2():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, sigma, step_size, loops):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.beta=1
		self.gamma=1
		self.loops=loops
		self.step_size=step_size
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


	def update_feature(self,x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def select_arm(self, time):
		self.s_list=np.zeros(self.item_num)
		temp=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		est_y_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		for i in range(self.item_num):
			x=self.item_feature[i]
			self.x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y_list[i]=np.dot(self.user_f, x)
			self.low_bound_list[i]=np.dot(self.user_f, x)-self.beta*np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.est_y_matrix[time, i]=np.dot(self.user_f, x)

		for j in range(self.item_num):
			self.s_list[j]=self.beta*self.x_norm_list[j]+est_y_list[j]
			# if self.s_list[j]>0:
			# 	temp[j]=1
			# else:
			# 	temp[j]=self.s_list[j]
			# print('beta, np.max(x_norm), np.max(est_y_list-est_y_list[j])', self.beta, np.max(self.x_norm_list),np.max(est_y_list-est_y_list[j]))

		# print('s_list', self.s_list)
		# self.s_list=temp 
		# print('s_list', np.round(self.s_list, decimals=2))
		soft_max=np.exp(self.gamma*self.s_list)/np.sum(np.exp(self.gamma*self.s_list))
		self.soft_max_matrix[time]=soft_max
		# print('soft_max', np.round(soft_max, decimals=2))
		ind=choice(range(self.item_num), p=soft_max)
		x=self.item_feature[ind]
		payoff=self.true_payoffs[ind]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[ind]
		return ind, x, payoff, regret

	def find_log_grad(self, time):
		a=self.gamma*np.dot(self.x_norm_list, np.exp(self.gamma*self.s_list))
		b=np.sum(np.exp(self.gamma*self.s_list))
		c=np.dot(self.s_list, np.exp(self.gamma*self.s_list))
		for i in range(self.item_num):
			self.beta_log_grad_matrix[time, i]=self.gamma*self.x_norm_list[i]-a/b+self.x_norm_list[i]
			self.gamma_log_grad_matrix[time, i]=self.s_list[i]-c/np.sum(np.exp(self.gamma*self.s_list))

	def update_gamma(self, time):
		gamma_grad=0
		for t in range(time-500, time):
			if t<0:
				pass
			else:
				temp_1=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[t], self.soft_max_matrix[t], self.gamma_log_grad_matrix[t])])
				gamma_grad+=temp_1
		# for t in range(self.iteration):
		# 	temp_1=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[t], self.soft_max_matrix[t], self.gamma_log_grad_matrix[t])])
		# 	gamma_grad+=temp_1
		self.gamma=self.gamma+self.step_size*gamma_grad

	def update_beta(self, time):
		beta_grad=0
		for t in range(time-500, time):
			if t<0:
				pass 
			else:
				temp_2=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[t], self.soft_max_matrix[t], self.beta_log_grad_matrix[t])])
				beta_grad+=temp_2
		# for t in range(self.iteration):
		# 	temp_2=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[t], self.soft_max_matrix[t], self.beta_log_grad_matrix[t])])
		# 	beta_grad+=temp_2
		self.beta=self.beta+self.step_size*beta_grad
		# if self.beta<0:
		# 	self.beta=0

	def generate_data(self):
		self.item_feature=Normalizer().fit_transform(np.random.normal(size=(self.item_num, self.dimension)))
		self.user_feature=np.random.normal(size=self.dimension)
		self.user_feature=self.user_feature/np.linalg.norm(self.user_feature)
		self.true_payoffs=np.dot(self.item_feature, self.user_feature)

	def run(self):
		cum_regret_loop=np.zeros(self.loops)
		beta_loop=np.zeros(self.loops)
		gamma_loop=np.zeros(self.loops)
		# for l in range(self.loops):
		error_list=np.zeros(self.iteration)
		cum_regret=[0]
		beta_list=np.zeros(self.iteration)
		gamma_list=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s ~~~~~ LinUCB-Soft'%(time, self.iteration))
			print('beta, gamma', np.round(self.beta), np.round(self.gamma))
			ind, x, y, regret=self.select_arm(time)
			self.update_feature(x, y)
			self.find_log_grad(time)
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=np.linalg.norm(self.user_f-self.user_feature)
			beta_list[time]=self.beta
			gamma_list[time]=self.gamma
			self.update_gamma(time)
			self.update_beta(time)
			# beta_loop[l]=self.beta
			# gamma_loop[l]=self.gamma

		return cum_regret[1:], error_list, beta_list, gamma_list, self.soft_max_matrix.T

		














