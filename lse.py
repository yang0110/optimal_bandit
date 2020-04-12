import numpy as np 

class LSE():
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
		self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(self.iteration/self.delta)))+np.sqrt(self.alpha)
		self.cov=self.alpha*np.identity(self.dimension)
		self.cov2=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.item_index=np.zeros(self.iteration)
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_matrix=np.zeros((self.item_num, self.iteration))
		self.low_ucb_matrix=np.zeros((self.item_num, self.iteration))
		self.remove=False
		self.item_index=[]
		self.payoff_error_matrix=np.zeros((self.item_num, self.iteration))
		self.worst_payoff_error=np.zeros(self.iteration)


	def select_arm(self):
		x_norm_list=np.zeros(self.item_num)
		cov_inv2=np.linalg.pinv(self.cov2)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv2),x))

		ind=np.argmax(x_norm_list)
		x=self.item_feature[ind]
		payoff=self.true_payoffs[ind]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-payoff
		return x, payoff, regret, ind

	def update_feature(self, x, y):
		self.cov+=np.outer(x, x)
		self.cov2+=np.outer(x, x)
		self.bias+=x*y 
		cov_inv=np.linalg.pinv(self.cov)
		self.user_f=np.dot(cov_inv, self.bias)

	def update_bounds(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.low_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			est_y=np.dot(self.user_f, x)
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv), x))
			self.upper_ucb_list[i]=est_y+self.beta*x_norm 
			self.low_ucb_list[i]=est_y-self.beta*x_norm
			self.upper_ucb_matrix[i,time]=est_y+self.beta*x_norm 
			self.low_ucb_matrix[i,time]=est_y-self.beta*x_norm
			self.payoff_error_matrix[i, time]=np.abs(self.true_payoffs[i]-est_y)

	def eliminate_arm(self):
		self.remove=False
		a=self.item_set.copy()
		for i in a:
			if np.max(self.low_ucb_list)>self.upper_ucb_list[i]:
				self.item_set.remove(i)
				self.remove=True 
			else:
				pass 

	def update_cov2(self):
		if self.remove==True:
			self.cov2=self.alpha*np.identity(self.dimension)
			# print('reset cov2')
		else:
			pass

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		item_index=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, remove=%s ~~~~~ LSE'%(time, self.iteration, len(self.item_set), self.remove))
			x,y, regret, index=self.select_arm()
			self.update_feature(x,y)
			self.update_bounds(time)
			self.eliminate_arm()
			self.update_cov2()
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
			item_index[time]=index
			self.worst_payoff_error[time]=np.max(self.payoff_error_matrix[:,time])

		return cum_regret, error, self.upper_ucb_matrix, self.low_ucb_matrix, self.payoff_error_matrix, self.worst_payoff_error



