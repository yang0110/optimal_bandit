import numpy as np 

class LSE():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, lambda_, combine_method):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.lambda_=lambda_
		self.combine_method=combine_method
		self.beta=0
		self.gamma=0
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.item_index=np.zeros(self.iteration)
		self.x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.remove=False
		self.remove_count=0
		self.remove_time_list=[]
		self.hist_est_y_dict={}
		self.hist_x_norm_dict={}
		self.hist_low_ucb_dict={}
		self.hist_upper_ucb_dict={}
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		self.avg_x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.avg_est_y_matrix=np.zeros((self.item_num, self.iteration))

	def initalize(self):
		for i in range(self.item_num):
			self.hist_est_y_dict[i]=[]
			self.hist_x_norm_dict[i]=[]
			self.hist_low_ucb_dict[i]=[]
			self.hist_upper_ucb_dict[i]=[]

	def update_beta(self, time):
		if time==0:
			pass 
		else:
			self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(time/self.delta)))+np.sqrt(self.alpha)
			# self.gamma=np.sqrt(2*np.log(1/self.delta))
			self.gamma=self.beta


	def select_arm(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		x_norm_list=np.zeros(self.item_num)
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			x_norm_list[i]=x_norm 
			self.x_norm_matrix[i,time]=x_norm 
			est_y=np.dot(self.user_f, x)
			self.current_x_norm_list[i]=x_norm 
			self.current_est_y_list[i]=est_y 

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

	def combine_method_1(self,time):
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			current_x_norm=self.current_x_norm_list[i]
			current_est_y=self.current_est_y_list[i]
			if self.remove_count==0:
				hist_x_norm=0
				hist_est_y=0
			else:
				hist_x_norm=self.hist_x_norm_dict[i]
				hist_est_y=self.hist_est_y_dict[i]

			avg_x_norm=self.gamma*np.mean(hist_x_norm)+self.lambda_*self.beta*current_x_norm
			avg_est_y=np.mean(hist_est_y)+self.lambda_*current_est_y

			self.low_ucb_list[i]=avg_est_y-avg_x_norm
			self.upper_ucb_list[i]=avg_est_y+avg_x_norm

			self.avg_est_y_matrix[i,time]=avg_est_y
			self.avg_x_norm_matrix[i,time]=avg_x_norm

	def combine_method_2(self, time):
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			current_x_norm=self.current_x_norm_list[i]
			current_est_y=self.current_est_y_list[i]
			if self.remove_count==0:
				hist_x_norm=0
				hist_est_y=0
			else:
				phase_index=np.argmin(np.array(self.hist_x_norm_dict[i]))
				hist_x_norm=self.hist_x_norm_dict[i][phase_index]
				hist_est_y=self.hist_est_y_dict[i][phase_index]

			avg_x_norm=self.gamma*hist_x_norm+self.lambda_*self.beta*current_x_norm
			avg_est_y=hist_est_y+self.lambda_*current_est_y

			self.low_ucb_list[i]=avg_est_y-avg_x_norm
			self.upper_ucb_list[i]=avg_est_y+avg_x_norm

			self.avg_est_y_matrix[i,time]=avg_est_y
			self.avg_x_norm_matrix[i,time]=avg_x_norm

	def combine_method_3(self, time):
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			current_x_norm=self.current_x_norm_list[i]
			current_est_y=self.current_est_y_list[i]
			avg_est_y=current_est_y
			avg_x_norm=current_x_norm

			self.low_ucb_list[i]=avg_est_y-self.beta*avg_x_norm
			self.upper_ucb_list[i]=avg_est_y+self.beta*avg_x_norm

			self.avg_est_y_matrix[i,time]=avg_est_y
			self.avg_x_norm_matrix[i,time]=avg_x_norm


	def eliminate_arm_function(self):
		self.remove=False
		for i in self.item_set:
			if np.max(self.low_ucb_list)>self.upper_ucb_list[i]:
				self.item_set.remove(i)
				self.remove=True 
			else:
				pass 

	def eliminate_arm(self,time):
		if self.combine_method==1:
			self.combine_method_1(time)
			self.eliminate_arm_function()

		elif self.combine_method==2:
			self.combine_method_2(time)
			self.eliminate_arm_function()

		else:
			self.combine_method_3(time)
			self.eliminate_arm_function()

	def reset(self, time):
		if self.remove==True:
			self.remove_count+=1
			self.remove_time_list.extend([time])
			for i in self.item_set:
				x=self.item_feature[i]
				hist_mean=np.dot(self.user_f, x)
				hist_x_norm=self.x_norm_matrix[i,time]
				self.hist_est_y_dict[i].extend([hist_mean])
				self.hist_x_norm_dict[i].extend([hist_x_norm])
				self.hist_low_ucb_dict[i].extend([hist_mean-self.beta*hist_x_norm])
				self.hist_upper_ucb_dict[i].extend([hist_mean+self.beta*hist_x_norm])

			self.cov=self.alpha*np.identity(self.dimension)
			self.bias=np.zeros(self.dimension)
			self.user_f=np.zeros(self.dimension)
		else:
			pass 

	def run(self, iteration):
		self.initalize()
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, item_num=%s, remove=%s ~~~~~ LSE'%(time, iteration, len(self.item_set), self.remove))
			self.update_beta(time)
			x,y,regret=self.select_arm(time)
			self.update_feature(x,y)
			self.eliminate_arm(time)
			self.reset(time)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret, error, self.item_index, self.x_norm_matrix, self.avg_x_norm_matrix














