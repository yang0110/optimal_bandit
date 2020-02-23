## Fix, finite arm set, the set set in each round
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
# os.chdir('C:/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from linucb_eli import LINUCB_ELI
from eliminator import ELI
from eli_test import ELI_TEST
from lse import LSE 
from lse_test import LSE_TEST
from se import SE
from se_test import SE_TEST
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=5
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
state=1 # small beta (exploitation), large beta(exploration), 1: true beta
lambda_=0.1


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)
worse_arm=np.argmin(true_payoffs)


linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, 1)

eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

se_model=SE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)

se_test_model=SE_TEST(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)


#####################

linucb_regret, linucb_error, linucb_item_index, linucb_x_norm_matrix, linucb_est_y_matrix, linucb_hist_low_matrix, linucb_hist_upper_matrix=linucb_model.run(iteration)

eli_regret, eli_error, eli_item_index, eli_x_norm_matrix, eli_est_y_matrix, eli_hist_low_matrix, eli_hist_upper_matrix=eli_model.run(iteration)


se_regret, se_error, se_item_index, se_x_norm_matrix, se_hist_low_matrix, se_hist_upper_matrix=se_model.run(iteration)

se_test_regret, se_test_error, se_test_item_index, se_test_x_norm_matrix, se_test_hist_low_matrix, se_test_hist_upper_matrix=se_test_model.run(iteration)

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, label='LinUCB')
plt.plot(eli_regret, label='Eliminator')
plt.plot(se_regret, label='SE')
plt.plot(se_test_regret, label='SE TEST')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LinUCB')
plt.plot(eli_error, label='Eliminator')
plt.plot(se_error, label='SE')
plt.plot(se_test_error, label='SE TEST')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.tight_layout()
plt.show()



x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='Paired', lut=None).colors


plt.figure(figsize=(8,6))
for i in range(item_num):
	#plt.plot(x, linucb_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
	plt.plot(x, linucb_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.1, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, linucb_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.ylim([-4,4])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('LinUCB: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'linucb_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(8,6))
for i in range(item_num):
	#plt.plot(x, eli_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
	plt.plot(x, se_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, se_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.ylim([-2,2])
plt.title('SE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'se_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(8,6))
for i in range(item_num):
	#plt.plot(x, eli_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
	plt.plot(x, se_test_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, se_test_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.ylim([-2,2])
plt.title('SE-TEST: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'se_test_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(8,6))
for i in range(item_num):
	#plt.plot(x, eli_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
	plt.plot(x, eli_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, eli_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
#plt.ylim([-4,4])
plt.title('Eliminator: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'eliminator_payoff_interval_each_arm'+'.png', dpi=100)
plt.show()




