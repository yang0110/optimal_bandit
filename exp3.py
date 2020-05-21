from scipy.special import softmax
import numpy as np 


class EXP3():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, gamma, eta):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.gamma=gamma
		self.eta=eta
		self.cov=self.alpha*np.identity(self.dimension)
		self.cov2=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.prob_list=np.zeros(self.item_num)
		self.uniform_prob_list=np.zeros(self.item_num)
		self.s_list=np.ones(self.item_num)
		self.s_prob_list=np.zeros(self.item_num)
		self.prob_matrix=np.zeros((self.iteration, self.item_num))
		self.item_hist={}
		self.est_y_list=np.zeros(self.item_num)

	def initial(self):
		for i in range(self.item_num):
			self.item_hist[i]=[]

	def uniform_prob(self):
		x_norm_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov2)
		for i in range(self.item_num):
			x=self.item_features[i]
			x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv), x))
		self.uniform_prob_list=softmax(x_norm_list)

	def select_arm(self, time):
		self.prob=self.eta*self.uniform_prob_list+(1-self.eta)*self.s_prob_list
		self.prob_matrix[time]=self.prob
		index=np.random.choice(range(self.item_num), p=self.prob)
		payoff=self.true_payoffs[index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		x=self.item_features[index]
		self.item_hist[index].extend([payoff])
		return index, regret

	def update_feature(self, index):
		x=self.item_features[index]
		self.cov2+=np.outer(x,x)
		length=len(self.item_hist[index])
		self.cov=length*np.outer(x,x)
		self.bias=x*np.sum(self.item_hist[index]) 
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		self.s_list[index]+=np.dot(self.user_f, x)

	def update_s_list(self, time):
		self.s_prob_list=softmax(self.gamma*self.s_list)

	def run(self):
		self.initial()
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, ~~~~~~~ EXP3'%(time, self.iteration))
			self.update_s_list(time)
			self.uniform_prob()
			index, regret=self.select_arm(time)
			self.update_feature(index)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error, self.prob_matrix.T



















