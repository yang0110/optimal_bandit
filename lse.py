import numpy as np 

class LSE():
	def __init__(self, dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, lambda_, gamma):
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
		self.beta=0
		self.gamma=gamma
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.item_index=np.zeros(self.iteration)
		self.current_x_norm_matrix=np.zeros((self.item_num, self.iteration))
		self.current_est_y_matrix=np.zeros((self.item_num, self.iteration))
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
		self.phase_length=np.zeros(self.item_num)
		self.average_weights=None
		self.hist_low_matrix=np.zeros((self.item_num, self.iteration))
		self.hist_upper_matrix=np.zeros((self.item_num, self.iteration))
		self.hist_user_f=[]
		self.est_y_matrix=np.zeros((self.item_num, self.iteration))

	def initalize(self):
		for i in range(self.item_num):
			self.hist_est_y_dict[i]=[]
			self.hist_x_norm_dict[i]=[]
			self.hist_low_ucb_dict[i]=[]
			self.hist_upper_ucb_dict[i]=[]

	def update_beta(self, time):
		#self.beta=2*self.sigma*np.sqrt(14*np.log(2*self.item_num*np.log2(self.iteration/self.delta)))+np.sqrt(self.alpha)
		self.beta=np.sqrt(2*np.log(1/self.delta))

	def select_arm(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		x_norm_list=np.zeros(self.item_num)
		self.current_x_norm_list=np.zeros(self.item_num)
		self.current_est_y_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			x_norm_list[i]=x_norm 
			est_y=np.dot(self.user_f, x)
			self.est_y_matrix[i,time]=est_y
			self.current_x_norm_list[i]=x_norm 
			self.current_est_y_list[i]=est_y 
			self.hist_low_matrix[i,time]=est_y-self.beta*x_norm 
			self.hist_upper_matrix[i,time]=est_y+self.beta*x_norm

		max_index=np.argmax(x_norm_list)
		x=self.item_feature[max_index]
		self.item_index[time]=max_index 
		payoff=self.true_payoffs[max_index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-payoff 
		return x, payoff, regret 

	def update_feature(self, x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		#self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		self.user_f+=self.gamma*(y-np.dot(self.user_f, x))*x

	def combine_method(self, time):
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			current_x_norm=self.current_x_norm_list[i]
			current_est_y=self.current_est_y_list[i]
			# if self.remove_count==0:
			# 	hist_x_norm=current_x_norm
			# 	hist_est_y=current_est_y
			# else:
			# 	hist_x_norm=self.hist_x_norm_dict[i]
			# 	hist_est_y=self.hist_est_y_dict[i]
			# 	hist_x_norm=np.average(hist_x_norm)
			# 	hist_est_y=np.average(hist_est_y)

			# avg_x_norm=current_x_norm+self.lambda_*hist_x_norm
			# #avg_x_norm=self.beta*self.lambda_*current_x_norm
			# avg_est_y=current_est_y+self.lambda_*hist_est_y

			self.low_ucb_list[i]=current_est_y-self.beta*current_x_norm
			self.upper_ucb_list[i]=current_est_y+self.beta*current_x_norm
			# self.hist_low_matrix[i,time]=self.low_ucb_list[i]
			# self.hist_upper_matrix[i,time]=self.upper_ucb_list[i]



	def eliminate_arm_function(self):
		self.remove=False
		a=self.item_set.copy()
		for i in a:
			if np.max(self.low_ucb_list)>self.upper_ucb_list[i]:
				self.item_set.remove(i)
				self.remove=True 
			else:
				pass 

	def eliminate_arm(self,time):
		self.combine_method(time)
		self.eliminate_arm_function()

	def reset(self, time):
		if self.remove==True:
			# self.remove_count+=1
			# self.remove_time_list.extend([time])
			# self.phase_length[self.remove_count-1]=time-self.phase_length[self.remove_count-1]
			# length_ratios=self.phase_length/np.sum(self.phase_length)
			# self.average_weights=length_ratios[1:self.remove_count+1]
			# for i in self.item_set:
			# 	x=self.item_feature[i]
			# 	hist_mean=np.dot(self.user_f, x)
			# 	hist_x_norm=self.current_x_norm_matrix[i,time]
			# 	self.hist_est_y_dict[i].extend([hist_mean])
			# 	self.hist_x_norm_dict[i].extend([hist_x_norm])
			# 	self.hist_low_ucb_dict[i].extend([hist_mean-self.beta*hist_x_norm])
			# 	self.hist_upper_ucb_dict[i].extend([hist_mean+self.beta*hist_x_norm])
			# self.hist_user_f.append(self.user_f)
			#print('self.hist_user_f', self.hist_user_f)
			self.cov=self.alpha*np.identity(self.dimension)
			#self.bias=np.zeros(self.dimension)
			#self.user_f=np.zeros(self.dimension)
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
		return cum_regret, error, self.item_index, self.est_y_matrix, self.hist_low_matrix, self.hist_upper_matrix














