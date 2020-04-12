import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
#os.chdir('C:/DATA/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from eliminator import ELI
from lse import LSE 
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=6
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=0.1 # regularizer


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)
worse_arm=np.argmin(true_payoffs)


linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)
#####################

linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error=linucb_model.run()

eli_regret, eli_error, eli_item_index, eli_upper_matrix, eli_low_matrix,eli_payoff_error_matrix, eli_worst_payoff_error=eli_model.run()

lse_regret, lse_error, lse_upper_matrix, lse_low_matrix, lse_payoff_error_matrix, lse_worst_payoff_error=lse_model.run()



plt.figure(figsize=(5,5))
plt.plot(linucb_regret, color='b', label='LinUCB')
plt.plot(eli_regret, color='y', label='SE')
plt.plot(lse_regret, color='r', label='LSE')
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, color='b', label='LinUCB')
plt.plot(eli_error, color='y', label='SE')
plt.plot(lse_error, color='r', label='LSE')
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.tight_layout()
plt.savefig(path+'error'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_worst_payoff_error, color='b', label='LinUCB')
plt.plot(eli_worst_payoff_error, color='y', label='SE')
plt.plot(lse_worst_payoff_error, color='r', label='LSE')
plt.ylim([-0.01,0.1])
plt.legend(loc=1, fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Worst payoff Error', fontsize=12)
plt.tight_layout()
plt.savefig(path+'worst_payoff_error'+'.png', dpi=100)
plt.show()


x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab20', lut=None).colors

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, lse_upper_matrix[i], '-', color=color_list[i], markevery=0.05, linewidth=2, markersize=5, label='Arm=%s'%(i))
	plt.plot(x, lse_low_matrix[i], '-.', color=color_list[i], markevery=0.01, linewidth=2, markersize=5)
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
	plt.plot(x, eli_upper_matrix[i], '-', color=color_list[i], markevery=0.05, linewidth=2, markersize=5, label='Arm=%s'%(i))
	plt.plot(x, eli_low_matrix[i], '-.', color=color_list[i], markevery=0.01, linewidth=2, markersize=5)

plt.legend(loc=1, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Interval', fontsize=12)
plt.title('SE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'se_payoff_interval'+'.png', dpi=200)
plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(x, linucb_upper_matrix[best_arm], label='LinUCB')
# plt.plot(x, eli_upper_matrix[best_arm], label='SE')
# plt.plot(x, lse_upper_matrix[best_arm], label='LSE')
# plt.legend(loc=0, fontsize=12)
# plt.title('Best Arm = %s'%(i), fontsize=12)
# plt.ylabel('UCB of Best Arm', fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(x, linucb_payoff_error_matrix[best_arm], label='LinUCB')
# plt.plot(x, eli_payoff_error_matrix[best_arm], label='SE')
# plt.plot(x, lse_payoff_error_matrix[best_arm], label='LSE')
# plt.legend(loc=0, fontsize=12)
# plt.title('Best Arm = %s'%(i), fontsize=12)
# plt.ylabel('Payoff Error of Best Arm', fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.show()


