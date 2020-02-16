import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
# os.chdir('C:/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from eliminator import ELI
from utils import *
path='../results/'
#np.random.seed(2018)


user_num=1
item_num=5
dimension=5
phase_num=8
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
state=1 # small beta (exploitation), large beta(exploration), 1: true beta


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)


eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

eli_regret, eli_error, eli_item_index, eli_x_norm_matrix, eli_est_y_matrix, eli_hist_low_matrix, eli_hist_upper_matrix=eli_model.run(iteration)


x=range(iteration)

color_list=matplotlib.cm.get_cmap(name='Paired', lut=None).colors

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, eli_est_y_matrix[i], color=color_list[i], linewidth=3, label='Best Arm=%s'%(i))
		plt.plot(x, eli_hist_upper_matrix[i], '-*', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)
		plt.plot(x, eli_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)
	else:
		plt.plot(x, eli_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
		plt.plot(x, eli_hist_upper_matrix[i], '-*', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)
		plt.plot(x, eli_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('Eliminator: Upper Bound, Mean, Lower Bound', fontsize=12)
plt.tight_layout()
plt.savefig(path+'eliminator_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()


