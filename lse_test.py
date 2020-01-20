import numpy as np 

class LSE_TEST():
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
		self.beta=0
		self.gamma=0
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.item_index=np.zeros(self.iteration)
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.remove=False
		self.remove_count=-1
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		self.low_ucb_matrix=np.zeros((self.item_num, self.iteration))
		self.upper_ucb_matrix=np.zeros((self.item_num, self.iteration))
		self.x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.phase_low_matrix=np.zeros((self.item_num, self.item_num))
		self.phase_upper_matrix=np.zeros((self.item_num, self.item_num))
		self.max_low=None
		self.min_upper=None


	def update_beta(self, time):
		if time==0:
			pass 
		else:
			self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(self.iteration/self.delta)))+np.sqrt(self.alpha)

	def select_arm(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		x_norm_list=np.zeros(self.item_num)
		self.low_ucb_list=np.ones(self.item_num)*(-10)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.x_norm_matrix[i,time]=x_norm
			x_norm_list[i]=x_norm 
			est_y=np.dot(self.user_f, x)
			self.current_x_norm_list[i]=x_norm 
			self.current_est_y_list[i]=est_y
			self.low_ucb_matrix[i,time]=est_y-self.beta*x_norm 
			self.upper_ucb_matrix[i,time]=est_y+self.beta*x_norm
			self.low_ucb_list[i]=est_y-self.beta*x_norm 
			self.upper_ucb_list[i]=est_y+self.beta*x_norm

		max_index=np.argmax(x_norm_list)
		x=self.item_feature[max_index]
		self.item_index[time]=max_index 
		payoff=self.true_payoffs[max_index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-payoff 
		return x, payoff, regret 


	def update_feature(self, x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)


	def eliminate_arm(self,time):
		self.remove=False
		for i in self.item_set:
			current_x_norm=self.current_x_norm_list[i]
			current_est_y=self.current_est_y_list[i]
			if self.remove_count==-1:
				self.max_low=np.max(self.low_ucb_list)
				self.min_upper=self.upper_ucb_list[i]
			else:
				a=self.phase_low_matrix[:, :self.remove_count+1]
				b=np.delete(a, i, axis=0)
				b_values=b.ravel()
				b_non_zeros=b_values[b_values!=0]
				self.max_low=np.max([np.max(b_non_zeros), np.max(self.low_ucb_list)])

				c=self.phase_upper_matrix[i].copy()
				d=c[c!=0]
				self.min_upper=np.min([np.min(d), self.upper_ucb_list[i]])

			if self.max_low>self.min_upper:
				self.item_set.remove(i)
				self.remove=True
			else:
				pass
		

	def reset(self, time):
		if self.remove==True:
			self.remove_count+=1
			for i in self.item_set:
				self.phase_low_matrix[i, self.remove_count]=self.low_ucb_matrix[i,time]
				self.phase_upper_matrix[i, self.remove_count]=self.upper_ucb_matrix[i,time]

			self.cov=self.alpha*np.identity(self.dimension)
			self.bias=np.zeros(self.dimension)
			self.user_f=np.zeros(self.dimension)
		else:
			pass 

	def run(self, iteration):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, remove=%s ~~~~~ LSE-TEST'%(time, iteration, len(self.item_set), self.remove))
			# print('self.phase_length', self.phase_length)
			# print('average_weights', self.average_weights)
			# print('self.hist_gap_dict', self.hist_gap_dict)
			self.update_beta(time)
			x,y,regret=self.select_arm(time)
			self.update_feature(x,y)
			self.eliminate_arm(time)
			self.reset(time)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret, error, self.low_ucb_matrix, self.upper_ucb_matrix, self.x_norm_matrix














