import numpy as np 

class M_ELI():
	def __init__(self, dimension, phase_num, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, lambda_):
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
		self.lambda_=lambda_
		self.beta=np.sqrt(2*np.log(1/self.delta))
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.lower_bound_list=np.zeros(self.item_num)
		self.upper_bound_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		self.current_x_norm_list=np.zeros(self.item_num)
		self.est_y_matrix=np.zeros((self.item_num, self.iteration))
		self.item_index=np.zeros(self.iteration)
		self.est_gaps=np.zeros((self.item_num, self.iteration))
		self.hist_est_y_dict={}
		self.hist_x_norm_dict={}
		self.hist_low_matrix=np.zeros((self.item_num, self.iteration))
		self.hist_upper_matrix=np.zeros((self.item_num, self.iteration))
	def initalize(self):
		for i in range(self.item_num):
			self.hist_est_y_dict[i]=[]
			self.hist_x_norm_dict[i]=[]

	def update_beta(self):
		self.beta=np.sqrt(2*np.log(1/self.delta))

	def select_arm(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.current_x_norm_list[i]=x_norm
			est_y=np.dot(self.user_f, x)
			self.current_est_y_list[i]=est_y

		max_index=np.argmax(self.current_x_norm_list)
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
		cov_inv=np.linalg.pinv(self.cov)
		for i in self.item_set:
			x=self.item_feature[i]
			self.hist_est_y_dict[i].extend([np.dot(self.user_f, x)])
			self.hist_x_norm_dict[i].extend([np.sqrt(np.dot(np.dot(x, cov_inv), x))])
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)

	def update_interval(self, time):
		self.lower_bound_list=np.zeros(self.item_num)
		self.upper_bound_list=np.zeros(self.item_num)
		for i in self.item_set:
			current_est_y=self.current_est_y_list[i]
			current_x_norm=self.current_x_norm_list[i]
			hist_est_y=np.average(self.hist_est_y_dict[i])
			hist_x_norm=np.average(self.hist_x_norm_dict[i])
			avg_est_y=current_est_y+self.lambda_*hist_est_y
			avg_x_norm=current_x_norm+self.lambda_*hist_x_norm
			self.lower_bound_list[i]=avg_est_y-self.beta*avg_x_norm
			self.upper_bound_list[i]=avg_est_y+self.beta*avg_x_norm
			self.hist_low_matrix[i,time]=self.lower_bound_list[i]
			self.hist_upper_matrix[i,time]=self.upper_bound_list[i]

	def eliminate_arm(self):
		a=self.item_set.copy()
		for j in a:
			if np.max(self.lower_bound_list)>self.upper_bound_list[j]:
				self.item_set.remove(j)
			else:
				pass 

	def run(self, iteration):
		self.initalize()
		cum_regret=[0]
		error=np.zeros(self.iteration)
		self.update_beta()
		for l in range(self.phase_num):
			start_time=2**l 
			end_time=2**(l+1)-1
			for time in range(start_time, end_time):
				t=time
				print('time/iteration=%s/%s, item_num=%s ~~~~ Modified Eliminator'%(time, iteration, len(self.item_set)))
				x,y, regret=self.select_arm(time)
				self.update_feature(x,y)
				self.update_interval(time)
				cum_regret.extend([cum_regret[-1]+regret])
				error[time]=np.linalg.norm(self.user_f-self.user_feature)
			self.eliminate_arm()
			self.reset()

		return cum_regret, error, self.item_index, self.hist_low_matrix, self.hist_upper_matrix



























