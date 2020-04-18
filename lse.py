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
		# self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(self.iteration/self.delta)))+np.sqrt(self.alpha)
		# self.beta=2*self.sigma*np.sqrt(14*np.log(2/self.delta))+np.sqrt(self.alpha)
		self.beta=np.sqrt(2*np.log(1/self.delta))+np.sqrt(self.alpha)
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
		self.noise_norm=np.zeros(self.iteration)
		self.noise_norm_phase=np.zeros(self.iteration)
		self.noise_bias=np.zeros(self.dimension)
		self.noise_bias_phase=np.zeros(self.dimension)
		self.bound=np.zeros(self.iteration)
		self.bound_phase=np.zeros(self.iteration)

	def update_error(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		cov_inv2=np.linalg.pinv(self.cov2)
		bound_list=np.zeros(self.item_num)
		phase_bound_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			bound_list[i]=self.beta*x_norm 
			x_norm_2=np.sqrt(np.dot(np.dot(x, cov_inv2),x))
			phase_bound_list[i]=(self.beta-np.sqrt(self.alpha))*x_norm_2
		self.bound[time]=np.max(bound_list)
		self.bound_phase[time]=np.max(phase_bound_list)


	def select_arm(self, time):
		x_norm_list=np.zeros(self.item_num)
		cov_inv2=np.linalg.pinv(self.cov2)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv2),x))

		ind=np.argmax(x_norm_list)
		x=self.item_feature[ind]
		noise=np.random.normal(scale=self.sigma)
		payoff=self.true_payoffs[ind]+noise
		regret=np.max(self.true_payoffs)-payoff
		x_best=self.item_feature[np.argmax(self.true_payoffs)]
		self.noise_bias+=x*noise
		self.noise_bias_phase+=x*noise
		self.noise_norm[time]=np.abs(np.dot(x_best, np.dot(np.linalg.pinv(self.cov), self.noise_bias)))
		self.noise_norm_phase[time]=np.abs(np.dot(x_best, np.dot(cov_inv2, self.noise_bias_phase)))
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
			self.noise_bias_phase=np.zeros(self.dimension)
			# print('reset cov2 and noise_bias_phase')
		else:
			pass

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		item_index=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, remove=%s ~~~~~ LSE'%(time, self.iteration, len(self.item_set), self.remove))
			x,y, regret, index=self.select_arm(time)
			self.update_feature(x,y)
			self.update_error(time)
			self.update_bounds(time)
			self.eliminate_arm()
			self.update_cov2()
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
			item_index[time]=index
			self.worst_payoff_error[time]=np.max(self.payoff_error_matrix[:,time])

		return cum_regret, error, self.upper_ucb_matrix, self.low_ucb_matrix, self.payoff_error_matrix, self.worst_payoff_error, self.noise_norm, self.noise_norm_phase, self.bound, self.bound_phase



