import numpy as np 

class LINUCB():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=np.sqrt(self.alpha)+np.sqrt(2*np.log(1/self.delta)+self.dimension*np.log(1+self.iteration/(self.dimension*self.alpha)))
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_index=np.zeros(self.iteration)
		self.low_matrix=np.zeros((self.item_num, self.iteration))
		self.upper_matrix=np.zeros((self.item_num, self.iteration))
		self.payoff_error_matrix=np.zeros((self.item_num, self.iteration))
		self.worst_payoff_error=np.zeros(self.iteration)


	def update_beta(self):
		self.beta=np.sqrt(self.alpha)+np.sqrt(2*np.log(1/self.delta)+self.dimension*np.log(1+self.iteration/(self.dimension*self.alpha)))

	def select_arm(self, time):
		index_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		for i in range(self.item_num):
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(self.user_f, x)
			index_list[i]=est_y+self.beta*x_norm
			self.low_matrix[i,time]=est_y-self.beta*x_norm 
			self.upper_matrix[i,time]=est_y+self.beta*x_norm
			self.payoff_error_matrix[i, time]=np.abs(self.true_payoffs[i]-est_y)

		max_index=np.argmax(index_list)
		self.item_index[time]=max_index
		x=self.item_feature[max_index]
		payoff=self.true_payoffs[max_index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-payoff
		return x, payoff, regret 

	def update_feature(self, x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time=%s/%s ~~~~~~LinUCB'%(time, self.iteration))
			x,y,regret=self.select_arm(time)
			self.update_feature(x,y)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
			self.worst_payoff_error[time]=np.max(self.payoff_error_matrix[:,time])
		return cum_regret[1:], error, self.item_index, self.upper_matrix, self.low_matrix, self.payoff_error_matrix, self.worst_payoff_error









