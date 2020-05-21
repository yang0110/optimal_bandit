import numpy as np 
from scipy.special import softmax
from numpy.random import choice
from sklearn.preprocessing import Normalizer, MinMaxScaler

class LSE_soft_base():
	def __init__(self, dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=None
		self.item_features=None
		self.true_payoffs=None
		self.alpha=alpha
		self.sigma=sigma
		self.weight1=weight1
		self.beta=beta
		self.gamma=gamma
		self.step_size_beta=step_size_beta
		self.step_size_gamma=step_size_gamma
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


	def initial(self):
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
		self.gamma=0

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
			self.est_y_matrix[time, i]=np.dot(self.user_f, x)

		for j in range(self.item_num):
			index=np.argmax(self.low_bound_list)
			self.x_norm_list2[j]=self.x_norm_list[j]+self.x_norm_list[index]
			self.s_list[j]=self.beta*(self.x_norm_list2[j])-(est_y_list[index]-est_y_list[j])


		# print('s_list', np.round(self.s_list, decimals=2))
		soft_max=np.exp(self.gamma*self.s_list)/np.sum(np.exp(self.gamma*self.s_list))
		self.soft_max_matrix[time]=soft_max
		# print('soft_max', np.round(soft_max, decimals=2))
		ind=choice(range(self.item_num), p=soft_max)
		x=self.item_features[ind]
		payoff=self.true_payoffs[ind]+np.random.normal(scale=self.sigma)
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


	def find_log_grad_gamma(self, time):
		c=np.dot(self.s_list, np.exp(self.gamma*self.s_list))
		for i in range(self.item_num):
			self.gamma_log_grad_matrix[time, i]=self.s_list[i]-c/np.sum(np.exp(self.gamma*self.s_list))

	def update_gamma_grad(self, time):
		temp_1=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.gamma_log_grad_matrix[time])])
		self.gamma_grad+=temp_1

	def update_gamma(self):
		self.gamma+=self.step_size_gamma*self.gamma_grad

	def update_beta_grad(self, time):
		temp_2=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.beta_log_grad_matrix[time], self.lagrange_grad_matrix[time])])
		self.beta_grad+=temp_2

	def update_beta(self, time):
		if (1/time)>0.01:
			pass
		else:
			self.step_size_beta=1/time
		self.beta+=self.step_size_beta*self.beta_grad

	def generate_data(self, item_num):
		self.item_features=Normalizer().fit_transform(np.random.normal(size=(self.item_num, self.dimension)))
		self.user_feature=np.random.normal(size=self.dimension)
		self.user_feature=self.user_feature/np.linalg.norm(self.user_feature)
		self.true_payoffs=np.dot(self.item_features, self.user_feature)

	def train(self, train_loops, item_num):
		self.cum_regret_loop=np.zeros(train_loops)
		self.beta_loop=np.zeros(train_loops)
		self.gamma_loop=np.zeros(train_loops)
		self.train_loops=train_loops
		for l in range(train_loops):
			self.generate_data(item_num)
			self.initial()
			error_list=np.zeros(self.iteration)
			cum_regret=[0]
			beta_list=np.zeros(self.iteration)
			gamma_list=np.zeros(self.iteration)
			old_payoff=0
			for time in range(self.iteration):
				print('Train-loop=%s, beta, gamma, time/iteration, %s/%s, %s, %s~~~~~ LSE-Soft-Base'%(l, time, self.iteration, np.round(self.beta, decimals=2), np.round(self.gamma, decimals=2) ))
				# print('beta, gamma', np.round(self.beta, decimals=2), np.round(self.gamma, decimals=2))
				ind, x, y, regret=self.select_arm(time, old_payoff)
				old_payoff=y
				self.update_feature(x, y)
				# self.find_log_grad_beta(time)
				# self.find_lagrange_grad(time)
				# self.update_beta_grad(time)
				self.find_log_grad_gamma(time)
				self.update_gamma_grad(time)
				cum_regret.extend([cum_regret[-1]+regret])
				error_list[time]=np.linalg.norm(self.user_f-self.user_feature)
				self.update_gamma()
			# self.update_beta(time)
			self.beta_loop[l]=self.beta
			self.gamma_loop[l]=self.gamma
			self.cum_regret_loop[l]=cum_regret[-1]
			
		return self.cum_regret_loop, self.beta_loop

	def run(self, user_feature, item_features, true_payoffs):
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.initial()
		error_list=np.zeros(self.iteration)
		cum_regret=[0]
		old_payoff=0
		for time in range(self.iteration):
			print('Test, time/iteration, %s/%s ~~~~~ LSE-Soft-Base'%(time, self.iteration))
			ind, x, y, regret=self.select_arm(time, old_payoff)
			old_payoff=y
			self.update_feature(x, y)
			self.find_log_grad_gamma(time)
			self.update_gamma_grad(time)
			self.update_gamma()
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=np.linalg.norm(self.user_f-self.user_feature)

		return cum_regret[1:], error_list, self.soft_max_matrix.T

		














