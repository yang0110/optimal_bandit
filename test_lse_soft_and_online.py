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
from lse_soft_online import LSE_soft_online
from lse_soft_base import LSE_soft_base
from lse_soft_v import LSE_soft_v
from linucb_soft import LinUCB_soft
from lints import LINTS
from linphe import LINPHE
from exp3 import EXP3
from giro import GIRO
from utils import *
path='../results/lse_soft_results/'
np.random.seed(2018)

user_num=1
item_num=10
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.1# noise
delta=0.1# high probability
alpha=0.1 # regularizer
step_size_beta=0.01
step_size_gamma=0.02
weight1=0.01
loop=1
train_loops=300
beta=3
beta_online=3
gamma=0
time_width=100


linucb_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
online_regret_matrix=np.zeros((loop, iteration))

# train model
lse_soft_model=LSE_soft(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

lse_soft_regret_list_train, lse_soft_beta_list_train=lse_soft_model.train(train_loops, item_num)

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
	online_model=LSE_soft_online(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta_online, gamma, time_width)

	#####################

	linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error, linucb_noise_norm, linucb_error_bound, linucb_threshold=linucb_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_prob_matrix, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)

	online_regret, online_error, online_prob_matrix, online_s_matrix, online_beta_list=online_model.run()


	linucb_regret_matrix[l]=linucb_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	online_regret_matrix[l]=online_regret


linucb_mean=np.mean(linucb_regret_matrix, axis=0)
linucb_std=linucb_regret_matrix.std(0)

lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)

online_mean=np.mean(online_regret_matrix, axis=0)
online_std=online_regret_matrix.std(0)


x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', color='b', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lse_soft_mean, '-p', color='c', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='c', alpha=0.2)
plt.plot(x, online_mean, '-o', color='k', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft-Online')
plt.fill_between(x, online_mean-online_std*0.95, online_mean+online_std*0.95, color='k', alpha=0.2)
plt.legend(loc=2, fontsize=12)
# plt.ylim([0,300])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret_shadow_soft'+'.png', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, color='b', label='LinUCB')
plt.plot(lse_soft_error, color='c', label='LSE-Soft')
plt.plot(online_error, color='k', label='LSe-Soft-Online')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lse_soft_regret_list_train, color='b')
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
# plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_training_regret_d_%s'%(dimension)+'.png', dpi=200)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lse_soft_beta_list_train, color='b')
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Beta', fontsize=12)
# plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_training_beta_d_%s'%(dimension)+'.png', dpi=200)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(online_beta_list, color='b')
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Beta', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_online_training_beta_d_%s'%(dimension)+'.png', dpi=200)
plt.show()


print(linucb_model.beta, lse_soft_model.beta, lse_soft_model.gamma)
x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='Set3', lut=None).colors


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_prob_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, lse_soft_prob_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-0.15,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_prob_matrix'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, online_prob_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, online_prob_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-0.15,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_online_prob_matrix'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, np.zeros(iteration),'-', color='k', linewidth=2)
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_s_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
	else:
		plt.plot(x, lse_soft_s_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-1,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('S(i,t)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_s_matrix'+'.png', dpi=300)
plt.show()

















