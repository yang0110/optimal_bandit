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
from lse_soft import LSE_soft
from lse_soft_base import LSE_soft_base
from lse_soft_v import LSE_soft_v
from linucb_soft import LinUCB_soft
from lints import LINTS
from utils import *
path='../results/lse_results/'
np.random.seed(2018)

user_num=1
item_num=10
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.1# noise
delta=0.01# high probability
alpha=0.1 # regularizer
step_size_beta=0.001
step_size_gamma=0.02
weight1=0.01
loop=1
train_loops=300
beta=1.69
gamma=0
epsilon=0.9
linucb_beta=np.sqrt(dimension*np.log(1+iteration/(alpha))+2*np.log(1/delta))

linucb_regret_matrix=np.zeros((loop, iteration))
lints_regret_matrix=np.zeros((loop, iteration))

lse_regret_matrix=np.zeros((loop, iteration))
eli_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
lse_soft_prob_matrix=np.zeros((item_num, iteration))

lse_soft_base_regret_matrix=np.zeros((loop, iteration))
lse_soft_base_prob_matrix=np.zeros((item_num, iteration))

# train model
lse_soft_model=LSE_soft(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

# lse_soft_regret_list_train, lse_soft_beta_list_train=lse_soft_model.train(train_loops, item_num)

# test data

for l in range(loop):

	item_features=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
	user_feature=np.random.normal(size=dimension)
	user_feature=user_feature/np.linalg.norm(user_feature)
	true_payoffs=np.dot(item_features, user_feature)
	best_arm=np.argmax(true_payoffs)
	worst_arm=np.argmin(true_payoffs)
	gaps=np.max(true_payoffs)-true_payoffs

	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, gaps)
	lints_model=LINTS(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, epsilon)
	eli_model=ELI(dimension, phase_num, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma)
	lse_model=LSE(dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, delta, sigma)
	lse_soft_base_model=LSE_soft_base(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1, linucb_beta, gamma)

	#####################

	linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error, linucb_noise_norm, linucb_error_bound, linucb_threshold=linucb_model.run()

	lints_regret, lints_error=lints_model.run()

	eli_regret, eli_error, eli_item_index, eli_upper_matrix, eli_low_matrix,eli_payoff_error_matrix, eli_worst_payoff_error, eli_noise_norm, eli_error_bound=eli_model.run()

	lse_regret, lse_error, lse_upper_matrix, lse_low_matrix, lse_payoff_error_matrix, lse_worst_payoff_error, lse_nosie_norm, lse_noise_norm_phase, lse_error_bound, lse_error_bound_phase, lse_threshold, lse_est_beta, lse_left_item_num, lse_est_beta2=lse_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_max, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)

	# lse_soft_base_regret, lse_soft_base_error, lse_soft_base_max=lse_soft_base_model.run(user_feature, item_features, true_payoffs)


	linucb_regret_matrix[l]=linucb_regret
	lints_regret_matrix[l]=lints_regret

	lse_regret_matrix[l]=lse_regret
	eli_regret_matrix[l]=eli_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	lse_soft_prob_matrix=lse_soft_max
	# lse_soft_base_regret_matrix[l]=lse_soft_base_regret
	# lse_soft_base_prob_matrix=lse_soft_base_max



linucb_mean=np.mean(linucb_regret_matrix, axis=0)
lints_mean=np.mean(lints_regret_matrix, axis=0)
lse_mean=np.mean(lse_regret_matrix, axis=0)
eli_mean=np.mean(eli_regret_matrix, axis=0)

linucb_std=linucb_regret_matrix.std(0)
lints_std=lints_regret_matrix.std(0)
lse_std=lse_regret_matrix.std(0)
eli_std=eli_regret_matrix.std(0)

lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)

# lse_soft_base_mean=np.mean(lse_soft_base_regret_matrix, axis=0)
# lse_soft_base_std=lse_soft_base_regret_matrix.std(0)

x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', color='b', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lints_mean, '-', color='m', markevery=0.1, linewidth=2, markersize=8, label='LinTS')
plt.fill_between(x, lints_mean-lints_std*0.95, lints_mean+lints_std*0.95, color='b', alpha=0.2)
plt.plot(x, eli_mean, '-o', color='r', markevery=0.1, linewidth=2, markersize=8, label='Successive Elimination')
plt.fill_between(x, eli_mean-eli_std*0.95, eli_mean+eli_std*0.95, color='r', alpha=0.2)
plt.plot(x, lse_mean, '-s', color='orange', markevery=0.1, linewidth=2, markersize=8, label='LSE')
plt.fill_between(x, lse_mean-lse_std*0.95, lse_mean+lse_std*0.95, color='orange', alpha=0.2)
plt.plot(x, lse_soft_mean, '-p', color='c', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='c', alpha=0.2)
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret_shadow'+'.png', dpi=300)
plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(x, linucb_mean, '-.', color='b', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
# plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
# plt.plot(x, eli_mean, '-o', color='r', markevery=0.1, linewidth=2, markersize=8, label='Successive Elimination')
# plt.fill_between(x, eli_mean-eli_std*0.95, eli_mean+eli_std*0.95, color='r', alpha=0.2)
# plt.plot(x, lse_mean, '-s', color='orange', markevery=0.1, linewidth=2, markersize=8, label='LSE')
# plt.fill_between(x, lse_mean-lse_std*0.95, lse_mean+lse_std*0.95, color='orange', alpha=0.2)
# plt.plot(x, lse_soft_mean, '-p', color='c', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
# plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='c', alpha=0.2)
# plt.plot(x, lse_soft_base_mean, '-|', color='m', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft-Base')
# plt.fill_between(x, lse_soft_base_mean-lse_soft_base_std*0.95, lse_soft_base_mean+lse_soft_base_std*0.95, color='m', alpha=0.2)
# plt.legend(loc=2, fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Cumulative Regret', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'regret_shadow2'+'.png', dpi=300)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_regret_list_train, color='b')
# plt.xlabel('Training iteration', fontsize=12)
# plt.ylabel('Cumulative Regret', fontsize=12)
# # plt.legend(loc=1, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_regret'+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_beta_list_train, color='b')
# plt.xlabel('Training iteration', fontsize=12)
# plt.ylabel('Beta', fontsize=12)
# # plt.legend(loc=1, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_beta'+'.png', dpi=200)
# plt.show()


print(linucb_model.beta, eli_model.beta, lse_model.beta, lse_soft_model.beta, lse_soft_model.gamma)
x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='Set3', lut=None).colors


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_prob_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, lse_soft_prob_matrix[i], '-', color=color_list[i], markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-0.15,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_prob_matrix'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, np.zeros(iteration),'-', color='k', linewidth=2)
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_s_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, lse_soft_s_matrix[i], '-', color=color_list[i], markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-1,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('S(i,t)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_s_matrix'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(x, np.zeros(iteration),'-', color='k', linewidth=2)
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_g_s_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, lse_soft_g_s_matrix[i], '-', color=color_list[i], markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-5,5])
plt.xlabel('Time', fontsize=12)
plt.ylabel('gamma*S(i,t)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_g_s_matrix'+'.png', dpi=300)
plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(x, lse_soft_base_prob_matrix[i], '-*', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
# 	else:
# 		plt.plot(x, lse_soft_base_prob_matrix[i], '-', color=color_list[i], markevery=0.2, linewidth=2, markersize=5)
# plt.legend(loc=4, fontsize=12)
# plt.ylim([-0.15,1.1])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_base_prob_matrix'+'.png', dpi=300)
# plt.show()

# plt.figure(figsize=(5,5))
# # for i in range(item_num):
# plt.plot(x, lse_est_beta[best_arm], '-', color='r', markevery=0.05, linewidth=2, markersize=5, label='est_beta1')
# plt.plot(x, lse_est_beta2[best_arm], '-', color='g', markevery=0.05, linewidth=2, markersize=5, label='est_beta2')
# # plt.plot(x, lse_left_item_num/(2*item_num),  color='k', label='item num')
# # plt.plot(x, lse_payoff_error_matrix[best_arm], color='b', label='payoff error')
# plt.legend(loc=4, fontsize=10)
# # plt.ylim([0,2])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('est beta', fontsize=12)
# plt.title('LSE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_est_beta'+'.png', dpi=300)
# plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_upper_matrix[i], '-', color='c', markevery=0.1, linewidth=2, markersize=5, label='Best Arm')
		plt.plot(x, lse_low_matrix[i], '-.', color='c', markevery=0.1, linewidth=2, markersize=5)
	else:
		plt.plot(x, lse_upper_matrix[i], '-', color=color_list[i], markevery=0.1, linewidth=2, markersize=5)
		# plt.plot(x, lse_low_matrix[i], '-.', color=color_list[i], markevery=0.1, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=10)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Estimated Reward Interval', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_payoff_interval'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, eli_upper_matrix[i], '-', color='c', markevery=0.1, linewidth=2, markersize=5, label='Best Arm')
		plt.plot(x, eli_low_matrix[i], '-.', color='c', markevery=0.1, linewidth=2, markersize=5)
	else:
		plt.plot(x, eli_upper_matrix[i], '-', color=color_list[i], markevery=0.1, linewidth=2, markersize=5)
		# plt.plot(x, eli_low_matrix[i], '-.', color=color_list[i], markevery=0.1, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=10)
plt.ylim([-3,3])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Estimated Reward Interval', fontsize=12)
plt.tight_layout()
plt.savefig(path+'eli_payoff_interval'+'.png', dpi=300)
plt.show()









