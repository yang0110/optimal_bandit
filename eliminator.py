import numpy as np 

class ELI():
	def __init__(self, dimension, phase_num, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma):
		self.dimension=dimension
		self.phase_num=phase_num
		self.iteration=2**phase_num
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.item_index=np.zeros(self.iteration)

	def update_beta(self, time):
		if time==0:
			pass 
		else:
			self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(time/self.delta)))+np.sqrt(self.alpha)

	def select_arm(self, time):
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		x_norm_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			x_norm_list[i]=x_norm
			self.x_norm_matrix[i,time]=x_norm 
			est_y=np.dot(self.user_f, x)
			self.low_ucb_list[i]=est_y-self.beta*x_norm 
			self.upper_ucb_list[i]=est_y+self.beta*x_norm

		max_index=np.argmax(x_norm_list)
		self.item_index[time]=max_index
		payoff=self.true_payoffs[max_index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[max_index]
		x=self.item_feature[max_index]
		return x, payoff, regret 

	def update_feature(self, x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)


	def reset(self):
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)

	def eliminate_arm(self):
		for i in self.item_set:
			if np.max(self.low_ucb_list)>self.upper_ucb_list[i]:
				self.item_set.remove(i)
			else:
				pass 
	def run(self, iteration):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for l in range(self.phase_num):
			start_time=2**l 
			end_time=2**(l+1)-1
			self.reset()
			for time in range(start_time, end_time):
				print('time/iteration=%s/%s, item_num=%s ~~~~ Eliminator'%(time, iteration, len(self.item_set)))
				self.update_beta(time)
				x,y, regret=self.select_arm(time)
				self.update_feature(x,y)
				self.eliminate_arm()
				cum_regret.extend([cum_regret[-1]+regret])
				error[time]=np.linalg.norm(self.user_f-self.user_feature)

		return cum_regret, error, self.item_index, self.x_norm_matrix



























