import numpy as np 

class LINUCB():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, state):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.state=state
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.item_index=np.zeros(self.iteration)

	def find_beta(self, time):
		# self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(time/self.delta)))+np.sqrt(self.alpha)
		self.beta=np.sqrt(self.alpha)+np.sqrt(2*np.log(1/self.delta)+self.dimension*np.log(1+self.iteration/(self.dimension*self.alpha)))
		self.beta=self.beta*self.state

	def select_arm(self, time):
		index_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		for i in range(self.item_num):
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y=np.dot(self.user_f, x)
			index_list[i]=est_y+self.beta*x_norm 
			self.x_norm_matrix[i,time]=x_norm

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

	def run(self, iteration):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(iteration):
			print('time=%s/%s ~~~~~~LinUCB'%(time, iteration))
			self.find_beta(time)
			x,y,regret=self.select_arm(time)
			self.update_feature(x,y)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error, self.item_index, self.x_norm_matrix









