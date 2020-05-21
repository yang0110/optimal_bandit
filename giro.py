import numpy as np 

class GIRO():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, pseudo_num):
		self.dimension=dimension
		self.iteration=iteration 
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.sigma=sigma
		self.pseudo_num=pseudo_num
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_count=np.zeros(self.item_num)
		self.item_hist={}
		self.est_y_list=np.ones(self.item_num)

	def initial(self):
		for i in range(self.item_num):
			self.item_hist[i]=[]

	def select_arm(self):
		index=np.argmax(self.est_y_list)
		payoff=self.true_payoffs[index]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[index]
		self.item_count[index]+=1
		self.item_hist[index].extend([payoff])
		self.item_hist[index].extend([0,1]*self.pseudo_num)
		return index, payoff, regret

	def update_feature(self, index, y):
		x=self.item_features[index]
		# self.cov+=(self.pseudo_num+1)*np.outer(x,x)
		# self.bias+=y*x+self.pseudo_num*x
		# self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		length=len(self.item_hist[index])
		samples=np.random.choice(self.item_hist[index], size=length)
		self.bias=x*np.sum(samples)
		self.cov=length*np.outer(x,x)+self.alpha*np.identity(self.dimension)
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)
		self.est_y_list[index]=np.dot(self.user_f, x)

	def run(self):
		self.initial()
		cum_regret=[0]
		error=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s, ~~~~~~~ Giro'%(time, self.iteration))
			index, payoff, regret=self.select_arm()
			self.update_feature(index, payoff)
			cum_regret.extend([cum_regret[-1]+regret])
			error[time]=np.linalg.norm(self.user_f-self.user_feature)
		return cum_regret[1:], error 



