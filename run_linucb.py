import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
# os.chdir('C:/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from utils import *
path='../results/'
#np.random.seed(2018)


user_num=1
item_num=5
dimension=5
iteration=200
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
state=1 # small beta (exploitation), large beta(exploration), 1: true beta


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)


linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, state)

linucb_regret, linucb_error, linucb_item_index, linucb_x_norm_matrix, linucb_est_y_matrix, linucb_hist_low_matrix, linucb_hist_upper_matrix=linucb_model.run(iteration)


x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='Paired', lut=None).colors

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, linucb_est_y_matrix[i], color=color_list[i], linewidth=3, label='Best Arm=%s'%(i))
		plt.plot(x, linucb_hist_upper_matrix[i], '-*', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)
		plt.plot(x, linucb_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)
	else:
		plt.plot(x, linucb_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
		plt.plot(x, linucb_hist_upper_matrix[i], '-*', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)
		plt.plot(x, linucb_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('LinUCB: Upper Bound, Mean, Lower Bound', fontsize=12)
plt.tight_layout()
plt.savefig(path+'linucb_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()


