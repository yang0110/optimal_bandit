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
from lse_soft_v import LSE_soft_v
from linucb_soft import LinUCB_soft
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=10
dimension=5
phase_num=11
iteration=2**phase_num
sigma=0.001# noise
delta=0.1# high probability
alpha=0.1 # regularizer
step_size_beta=0.001
step_size_gamma=0.001
weight1=0.1
loop=5
train_loops=20


linucb_regret_matrix=np.zeros((loop, iteration))
lse_regret_matrix=np.zeros((loop, iteration))
eli_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
lse_soft_beta_matrix=np.zeros((loop, iteration))
lse_soft_gamma_matrix=np.zeros((loop, iteration))
lse_soft_prob_matrix=np.zeros((loop, iteration))

lse_soft_v_regret_matrix=np.zeros((loop, iteration))
# lse_soft_v_beta_matrix=np.zeros((loop, iteration))
# lse_soft_v_gamma_matrix=np.zeros((loop, iteration))
lse_soft_v_prob_matrix=np.zeros((loop, iteration))


lse_soft_model=LSE_soft(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1)

lse_soft_model.train(train_loops)

lse_soft_v_model=LSE_soft_v(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1)

lse_soft_v_model.train(train_loops)


for l in range(loop):

	item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
	user_feature=np.random.normal(size=dimension)
	user_feature=user_feature/np.linalg.norm(user_feature)
	true_payoffs=np.dot(item_feature, user_feature)
	best_arm=np.argmax(true_payoffs)
	worst_arm=np.argmin(true_payoffs)
	gaps=np.max(true_payoffs)-true_payoffs

	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, gaps)

	eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

	lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)

	#####################

	linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error, linucb_noise_norm, linucb_error_bound, linucb_threshold=linucb_model.run()

	eli_regret, eli_error, eli_item_index, eli_upper_matrix, eli_low_matrix,eli_payoff_error_matrix, eli_worst_payoff_error, eli_noise_norm, eli_error_bound=eli_model.run()

	lse_regret, lse_error, lse_upper_matrix, lse_low_matrix, lse_payoff_error_matrix, lse_worst_payoff_error, lse_nosie_norm, lse_noise_norm_phase, lse_error_bound, lse_error_bound_phase, lse_threshold, lse_est_beta, lse_left_item_num, lse_est_beta2=lse_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_beta, lse_soft_gamma, lse_soft_regret_loop, lse_soft_max=lse_soft_model.run(user_feature, item_feature, true_payoffs)

	lse_soft_v_regret, lse_soft_v_error, lse_soft_v_beta, lse_soft_v_gamma, lse_soft_v_regret_loop, lse_soft_v_max=lse_soft_v_model.run(user_feature, item_feature, true_payoffs)


	linucb_regret_matrix[l]=linucb_regret
	lse_regret_matrix[l]=lse_regret
	eli_regret_matrix[l]=eli_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	# lse_soft_beta_matrix[l]=lse_soft_beta
	# lse_soft_gamma_matrix[l]=lse_soft_gamma
	lse_soft_prob_matrix[l]=lse_soft_max[best_arm]

	lse_soft_v_regret_matrix[l]=lse_soft_v_regret
	# lse_soft_v_beta_matrix[l]=lse_soft_v_beta
	# lse_soft_v_gamma_matrix[l]=lse_soft_v_gamma
	lse_soft_v_prob_matrix[l]=lse_soft_v_max[best_arm]

	# linucb_soft_regret_matrix[l]=linucb_soft_regret
	# linucb_soft_beta_matrix[l]=linucb_soft_beta
	# linucb_soft_gamma_matrix[l]=linucb_soft_gamma
	# linucb_soft_prob_matrix[l]=linucb_soft_max[best_arm]

linucb_mean=np.mean(linucb_regret_matrix, axis=0)
lse_mean=np.mean(lse_regret_matrix, axis=0)
eli_mean=np.mean(eli_regret_matrix, axis=0)
linucb_std=linucb_regret_matrix.std(0)
lse_std=lse_regret_matrix.std(0)
eli_std=eli_regret_matrix.std(0)
lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)

lse_soft_v_mean=np.mean(lse_soft_v_regret_matrix, axis=0)
lse_soft_v_std=lse_soft_v_regret_matrix.std(0)
# linucb_soft_mean=np.mean(linucb_soft_regret_matrix, axis=0)
# linucb_soft_std=linucb_soft_regret_matrix.std(0)


x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)

plt.plot(x, lse_mean, '-s', markevery=0.1, linewidth=2, markersize=8, label='LSE')
plt.fill_between(x, lse_mean-lse_std*0.95, lse_mean+lse_std*0.95, color='r', alpha=0.2)

plt.plot(x, eli_mean, '-o', markevery=0.1, linewidth=2, markersize=8, label='Successive Elimination')
plt.fill_between(x, eli_mean-eli_std*0.95, eli_mean+eli_std*0.95, color='y', alpha=0.2)

plt.plot(x, lse_soft_mean, '-p', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='g', alpha=0.2)

plt.plot(x, lse_soft_v_mean, '-|', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft-V')
plt.fill_between(x, lse_soft_v_mean-lse_soft_v_std*0.95, lse_soft_v_mean+lse_soft_v_std*0.95, color='k', alpha=0.2)
# plt.plot(x, linucb_soft_mean, '-|', color='k', markevery=0.1, linewidth=2, markersize=8, label='LinUCB-Soft')
# plt.fill_between(x, linucb_soft_mean-linucb_soft_std*0.95, linucb_soft_mean+linucb_soft_std*0.95, color='k', alpha=0.2)
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret_shadow2'+'.png', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, 'b', linewidth=2, label='LinUCB')
plt.plot(lse_regret, 'r', linewidth=2,label='LSE')
plt.plot(eli_regret, 'y', linewidth=2,label='Successive Elimination')
plt.plot(lse_soft_regret, 'g', linewidth=2,label='LSE-Soft')
plt.plot(lse_soft_v_regret, 'k', linewidth=2,label='LSE-Soft-V')
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(lse_soft_regret_loop)
plt.xlabel('loops')
plt.ylabel('regret')
plt.title('LSE-Soft')
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lse_soft_v_regret_loop)
plt.xlabel('loops')
plt.ylabel('regret')
plt.title('LSE-Soft-V')
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lse_soft_beta)
plt.xlabel('loops')
plt.ylabel('beta')
plt.title('LSE-Soft')
plt.show()



# beta_mean=np.mean(lse_soft_beta_matrix, axis=0)
# beta_std=lse_soft_beta_matrix.std(0)
# gamma_mean=np.mean(lse_soft_gamma_matrix, axis=0)
# gamma_std=lse_soft_gamma_matrix.std(0)
# prob_mean=np.mean(lse_soft_prob_matrix, axis=0)
# prob_std=lse_soft_prob_matrix.std(0)

# plt.figure(figsize=(5,5))
# plt.plot(x, beta_mean, '-', markevery=0.1, linewidth=2, markersize=8)
# plt.fill_between(x, beta_mean-beta_std*0.95, beta_mean+beta_std*0.95, color='b', alpha=0.2)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('beta', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_beta'+'.png', dpi=300)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(x, gamma_mean, '-', markevery=0.1, linewidth=2, markersize=8)
# plt.fill_between(x, gamma_mean-gamma_std*0.95, gamma_mean+gamma_std*0.95, color='y', alpha=0.2)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('gamma', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_gamma'+'.png', dpi=300)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(x, prob_mean, '-', markevery=0.1, linewidth=2, markersize=8)
# plt.fill_between(x, prob_mean-prob_std*0.95, prob_mean+prob_std*0.95, color='r', alpha=0.2)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability of best arm', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_prob'+'.png', dpi=300)
# plt.show()

print(linucb_model.beta, eli_model.beta, lse_model.beta, lse_soft_model.beta, lse_soft_model.gamma)
x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab10', lut=None).colors


plt.figure(figsize=(5,5))
plt.plot(x, lse_soft_max[best_arm], '-.', color='r', markevery=0.2, linewidth=2, markersize=5, label='Best-arm=%s'%(best_arm))
plt.plot(x, lse_soft_max[worst_arm], '-.', color='b', markevery=0.2, linewidth=2, markersize=5, label='Worst-arm=%s'%(worst_arm))
plt.legend(loc=4, fontsize=10)
plt.ylim([-0.1,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability of arms', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_prob_best_arm_worst_arm'+'.png', dpi=300)
plt.show()




plt.figure(figsize=(5,5))
plt.plot(x, lse_soft_v_max[best_arm], '-.', color='r', markevery=0.2, linewidth=2, markersize=5, label='Best-arm=%s'%(best_arm))
plt.plot(x, lse_soft_v_max[worst_arm], '-.', color='b', markevery=0.2, linewidth=2, markersize=5, label='Worst-arm=%s'%(worst_arm))
plt.legend(loc=4, fontsize=10)
plt.ylim([-0.1,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability of arms', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_v_prob_best_arm_worst_arm'+'.png', dpi=300)
plt.show()
# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	plt.plot(x, lse_soft_max[i], '-.', color=color_list[i], markevery=0.2, linewidth=2, markersize=5, label='Arm=%s'%(i))
# plt.legend(loc=4, fontsize=10)
# plt.ylim([-0.1,1.1])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability of arms', fontsize=12)
# plt.title('Best Arm=%s'%(best_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_prob_matrix'+'.png', dpi=300)
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	plt.plot(x, linucb_soft_max[i], '-.', color=color_list[i], markevery=0.2, linewidth=2, markersize=5, label='Arm=%s'%(i))
# plt.legend(loc=4, fontsize=10)
# plt.ylim([-0.1,1.1])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability of arms', fontsize=12)
# # plt.title('Best Arm=%s'%(best_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'linucb_soft_prob_matrix'+'.png', dpi=300)
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


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	plt.plot(x, lse_upper_matrix[i], '-', color=color_list[i], markevery=0.05, linewidth=2, markersize=5, label='Arm=%s'%(i))
# 	plt.plot(x, lse_low_matrix[i], '-.', color=color_list[i], markevery=0.01, linewidth=2, markersize=5)
# plt.legend(loc=4, fontsize=10)
# plt.ylim([-2,2])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Estimated Reward Interval', fontsize=12)
# # plt.title('LSE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_payoff_interval'+'.png', dpi=300)
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	plt.plot(x, eli_upper_matrix[i], '-', color=color_list[i], markevery=0.05, linewidth=2, markersize=5, label='Arm=%s'%(i))
# 	plt.plot(x, eli_low_matrix[i], '-.', color=color_list[i], markevery=0.01, linewidth=2, markersize=5)
# # plt.legend(loc=4, fontsize=10)
# plt.ylim([-2,2])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Estimated Reward Interval', fontsize=12)
# # plt.title('LSE: Best Arm=%s, Worst Arm=%s'%(best_arm, worse_arm), fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'eli_payoff_interval'+'.png', dpi=300)
# plt.show()










