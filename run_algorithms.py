import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
#os.chdir('C:/DATA/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from linucb_eli import LINUCB_ELI
from eliminator import ELI
from m_eli import M_ELI
from eli_test import ELI_TEST
from lse import LSE 
from lse_test import LSE_TEST
from se import SE
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
lambda_=0.5
gamma=0.1


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)
worse_arm=np.argmin(true_payoffs)


linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, 1)

eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, gamma)

se_model=SE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, gamma)

lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, lambda_, gamma)

m_eli_model=M_ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, gamma)
#####################

linucb_regret, linucb_error, linucb_item_index, linucb_x_norm_matrix, linucb_est_y_matrix, linucb_hist_low_matrix, linucb_hist_upper_matrix=linucb_model.run(iteration)

eli_regret, eli_error, eli_item_index, eli_x_norm_matrix, eli_est_y_matrix, eli_hist_low_matrix, eli_hist_upper_matrix, eli_item_num_list=eli_model.run(iteration)

se_regret, se_error, se_item_index, se_x_norm_matrix, se_est_y_matrix, se_hist_low_matrix, se_hist_upper_matrix, se_item_num_list=se_model.run(iteration)

lse_regret, lse_error, lse_item_index, lse_est_y_matrix, lse_hist_low_matrix, lse_hist_upper_matrix, lse_item_num_list=lse_model.run(iteration)

m_eli_regret, m_eli_error, m_eli_item_index, m_eli_x_norm_matrix, m_eli_est_y_matrix,  m_eli_hist_low_matrix, m_eli_hist_upper_matrix, m_eli_item_num_list=m_eli_model.run(iteration)

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, label='LinUCB')
plt.plot(eli_regret, label='Eliminator')
plt.plot(m_eli_regret, label='M-Eliminator')
plt.plot(se_regret, label='SE')
plt.plot(lse_regret, label='LSE')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LinUCB')
plt.plot(eli_error, label='Eliminator')
plt.plot(m_eli_error, label='M-Eliminator')
plt.plot(se_error, label='SE')
plt.plot(lse_error, label='LSE')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.tight_layout()
plt.savefig(path+'error'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(eli_item_num_list, label='Eliminator')
plt.plot(m_eli_item_num_list, label='M-Eliminator')
plt.plot(se_item_num_list, label='SE')
plt.plot(lse_item_num_list, label='LSE')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Item num', fontsize=12)
plt.tight_layout()
plt.savefig(path+'item_num'+'.png', dpi=100)
plt.show()

beta=np.sqrt(2*np.log(1/delta))
x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab20', lut=None).colors


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	#plt.plot(x, linucb_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
# 	plt.plot(x, linucb_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.1, linewidth=2, markersize=8, label='Arm=%s'%(i))
# 	plt.plot(x, linucb_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.1, linewidth=2, markersize=8)

# plt.legend(loc=1, fontsize=10)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Payoff Interval', fontsize=12)
# plt.title('LinUCB: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'linucb_payoff_interval_each_arm'+'.png', dpi=200)
# plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, se_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, se_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('SE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'se_payoff_interval'+'.png', dpi=200)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, lse_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, lse_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)
plt.legend(loc=1, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('LSE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_payoff_interval'+'.png', dpi=200)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, eli_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, eli_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)

plt.legend(loc=1, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('Eliminator: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'eli_payoff_interval'+'.png', dpi=200)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, m_eli_hist_upper_matrix[i], '-.', color=color_list[i], markevery=0.05, linewidth=2, markersize=8, label='Arm=%s'%(i))
	plt.plot(x, m_eli_hist_low_matrix[i], '-|', color=color_list[i], markevery=0.05, linewidth=2, markersize=8)
plt.legend(loc=1, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('M-Eliminator: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'m_eli_payoff_interval'+'.png', dpi=200)
plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	plt.plot(x, linucb_est_y_matrix[i], color=color_list[i], linewidth=3, label='Arm=%s'%(i))
# plt.legend(loc=1, fontsize=10)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Payoff', fontsize=12)
# plt.title('LinUCB: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'linucb_payoff'+'.png', dpi=200)
# plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, eli_est_y_matrix[i],'-.', color=color_list[i], linewidth=2, label='Arm=%s'%(i))
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff', fontsize=12)
plt.title('Eliminator: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'eli_payoff'+'.png', dpi=200)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, m_eli_est_y_matrix[i],'-.', color=color_list[i], linewidth=2, label='Arm=%s'%(i))
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff', fontsize=12)
plt.title('M-Eliminator: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'m_eli_payoff'+'.png', dpi=200)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, se_est_y_matrix[i],'-.', color=color_list[i], linewidth=2, label='Arm=%s'%(i))
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff', fontsize=12)
plt.title('SE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'se_payoff'+'.png', dpi=200)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, lse_est_y_matrix[i],'-.', color=color_list[i], linewidth=2, label='Arm=%s'%(i))
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff', fontsize=12)
plt.title('LSE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_payoff'+'.png', dpi=200)
plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(se_item_index, '.', label='SE')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Selected Arm', fontsize=12)
# plt.title('SE: Best Arm=%s, Worse arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'se_item_index'+'.png', dpi=200)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(linucb_item_index,'.', label='LinUCB')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Selected Arm', fontsize=12)
# plt.title('LinUCB: Best Arm=%s, Worse arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'linucb_item_index'+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_item_index, '.', label='LSE')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Selected Arm', fontsize=12)
# plt.title('LSE: Best Arm=%s, Worse arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_item_index'+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(eli_item_index,'.', label='Eliminator')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Selected Arm', fontsize=12)
# plt.title('Eliminiator: Best Arm=%s, Worse arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'eliminator_item_index'+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(m_eli_item_index,'.', label='Eliminator')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Selected Arm', fontsize=12)
# plt.title('M-Eliminiator: Best Arm=%s, Worse arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'m_eliminator_item_index'+'.png', dpi=200)
# plt.show()


