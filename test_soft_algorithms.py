
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
from lse_soft2 import LSE_soft2
from linucb_soft2 import LinUCB_soft2
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=10
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=0.1 # regularizer
step_size_beta=0.01
step_size_gamma=0.01
weight1=-1
loop=1
lse_soft_loops=1


linucb_regret_matrix=np.zeros((loop, iteration))
lse_regret_matrix=np.zeros((loop, iteration))
eli_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
lse_soft_beta_matrix=np.zeros((loop, iteration))
lse_soft_gamma_matrix=np.zeros((loop, iteration))
lse_soft_prob_best_arm_matrix=np.zeros((loop, iteration))
lse_soft_prob_worst_arm_matrix=np.zeros((loop, iteration))


linucb_soft_regret_matrix=np.zeros((loop, iteration))
linucb_soft_beta_matrix=np.zeros((loop, iteration))
linucb_soft_gamma_matrix=np.zeros((loop, iteration))
linucb_soft_prob_best_arm_matrix=np.zeros((loop, iteration))
linucb_soft_prob_worst_arm_matrix=np.zeros((loop, iteration))

for l in range(loop):

	item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
	user_feature=np.random.normal(size=dimension)
	user_feature=user_feature/np.linalg.norm(user_feature)
	true_payoffs=np.dot(item_feature, user_feature)
	best_arm=np.argmax(true_payoffs)
	worst_arm=np.argmin(true_payoffs)
	gaps=np.max(true_payoffs)-true_payoffs


	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, gaps)

	lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)

	lse_soft_model=LSE_soft2(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, sigma, step_size_beta, step_size_gamma, weight1, lse_soft_loops)

	# linucb_soft_model=LinUCB_soft(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, sigma, step_size, lse_soft_loops)

	#####################

	linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error, linucb_noise_norm, linucb_error_bound, linucb_threshold=linucb_model.run()


	lse_regret, lse_error, lse_upper_matrix, lse_low_matrix, lse_payoff_error_matrix, lse_worst_payoff_error, lse_nosie_norm, lse_noise_norm_phase, lse_error_bound, lse_error_bound_phase, lse_threshold, lse_est_beta, lse_left_item_num, lse_est_beta2=lse_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_beta, lse_soft_gamma, lse_soft_max=lse_soft_model.run()

	# linucb_soft_regret, linucb_soft_error, linucb_soft_beta, linucb_soft_gamma, linucb_soft_max=linucb_soft_model.run()

	linucb_regret_matrix[l]=linucb_regret
	lse_regret_matrix[l]=lse_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	lse_soft_beta_matrix[l]=lse_soft_beta
	lse_soft_gamma_matrix[l]=lse_soft_gamma
	lse_soft_prob_best_arm_matrix[l]=lse_soft_max[best_arm]
	lse_soft_prob_worst_arm_matrix[l]=lse_soft_max[worst_arm]

linucb_mean=np.mean(linucb_regret_matrix, axis=0)
lse_mean=np.mean(lse_regret_matrix, axis=0)
linucb_std=linucb_regret_matrix.std(0)
lse_std=lse_regret_matrix.std(0)
lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)


x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lse_mean, '-s', color='orange', markevery=0.1, linewidth=2, markersize=8, label='LSE')
plt.fill_between(x, lse_mean-lse_std*0.95, lse_mean+lse_std*0.95, color='orange', alpha=0.2)
plt.plot(x, lse_soft_mean, '-p', color='g', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='g', alpha=0.2)
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret_shadow2'+'.png', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, 'b', linewidth=2, label='LinUCB')
plt.plot(lse_regret, 'r', linewidth=2,label='LSE')
plt.plot(lse_soft_regret, 'g', linewidth=2,label='LSE-Soft')
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret'+'.png', dpi=300)
plt.show()


beta_mean=np.mean(lse_soft_beta_matrix, axis=0)
beta_std=lse_soft_beta_matrix.std(0)
gamma_mean=np.mean(lse_soft_gamma_matrix, axis=0)
gamma_std=lse_soft_gamma_matrix.std(0)
prob_best_arm_mean=np.mean(lse_soft_prob_best_arm_matrix, axis=0)
prob_best_arm_std=lse_soft_prob_best_arm_matrix.std(0)
prob_worst_arm_mean=np.mean(lse_soft_prob_worst_arm_matrix, axis=0)
prob_worst_arm_std=lse_soft_prob_worst_arm_matrix.std(0)


plt.figure(figsize=(5,5))
plt.plot(x, beta_mean, '-', markevery=0.1, linewidth=2, markersize=8)
plt.fill_between(x, beta_mean-beta_std*0.95, beta_mean+beta_std*0.95, color='b', alpha=0.2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('beta', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_beta'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, gamma_mean, '-', markevery=0.1, linewidth=2, markersize=8)
plt.fill_between(x, gamma_mean-gamma_std*0.95, gamma_mean+gamma_std*0.95, color='y', alpha=0.2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('gamma', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_gamma'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(x, prob_best_arm_mean, '-', color='r', markevery=0.1, linewidth=2, markersize=8, label='best arm=%s'%(best_arm))
plt.fill_between(x, prob_best_arm_mean-prob_best_arm_std*0.95, prob_best_arm_mean+prob_best_arm_std*0.95, color='r', alpha=0.2)
plt.plot(x, prob_worst_arm_mean, '-', color='b', markevery=0.1, linewidth=2, markersize=8, label='worst arm=%s'%(worst_arm))
plt.fill_between(x, prob_worst_arm_mean-prob_worst_arm_std*0.95, prob_worst_arm_mean+prob_worst_arm_std*0.95, color='b', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability of best arm', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_prob_best_worst_arm'+'.png', dpi=300)
plt.show()



print(linucb_model.beta,lse_model.beta, lse_soft_model.beta, lse_soft_model.gamma)
x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab10', lut=None).colors


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(x, lse_soft_max[i], '-.', color=color_list[i], markevery=0.2, linewidth=2, markersize=5, label='Arm=%s'%(i))
plt.legend(loc=4, fontsize=10)
plt.ylim([-0.1,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability of arms', fontsize=12)
# plt.title('Best Arm=%s'%(best_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_prob_matrix'+'.png', dpi=300)
plt.show()












