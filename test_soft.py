import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
# os.chdir('C:/DATA/Kaige_Research/Code/optimal_bandit/code/')
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
item_num=20
dimension=10
phase_num=9
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.01
step_size_gamma=0.02
weight1=0.01
loop=1
train_loops=100
beta=5
gamma=0
loop_num=10


user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)

beta_matrix=np.zeros((loop_num, train_loops))
regret_matrix=np.zeros((loop_num, train_loops))
for l in range(loop_num):

	lse_soft_model=LSE_soft(dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)
	lse_soft_regret_list_train, lse_soft_beta_list_train, lse_soft_prob_matrix, lse_soft_beta_gradient=lse_soft_model.train(train_loops, item_num)
	beta_matrix[l]=lse_soft_beta_list_train
	regret_matrix[l]=lse_soft_regret_list_train


beta_mean=np.mean(beta_matrix, axis=0)
beta_std=beta_matrix.std(0)
regret_mean=np.mean(regret_matrix, axis=0)
regret_std=regret_matrix.std(0)


np.save(path+'lse_soft_offline_beta_mean_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), beta_mean)
np.save(path+'lse_soft_offline_beta_std_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), beta_std)
np.save(path+'lse_soft_offline_regret_mean_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), regret_mean)
np.save(path+'lse_soft_offline_regret_std_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), regret_std)


plt.figure(figsize=(5,5))
plt.plot(lse_soft_beta_gradient)
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Gradient', fontsize=12)
plt.show()

x=range(train_loops)
plt.figure(figsize=(5,5))
plt.plot(x, beta_mean, color='y', linewidth=2, label='d = %s'%(dimension))
plt.fill_between(x, beta_mean-beta_std*0.95, beta_mean+beta_std*0.95, color='y', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'lse_soft_offline_beta_d'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, regret_mean, color='r', linewidth=2, label='d = %s'%(dimension))
plt.fill_between(x, regret_mean-regret_std*0.95, regret_mean+regret_std*0.95, color='r', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'lse_soft_offline_beta_d'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(lse_soft_prob_matrix[i], markevery=0.2, linewidth=2)
plt.ylim([-0.15,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.tight_layout()
plt.show()


# np.save(path+'lse_soft_offline_beta_mean_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), beta_mean)
# np.save(path+'lse_soft_offline_beta_std_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), beta_std)
# np.save(path+'lse_soft_offline_regret_mean_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), regret_mean)
# np.save(path+'lse_soft_offline_regret_std_item_%s_d_%s_t_%s.npy'%(item_num, dimension, phase_num), regret_std)




beta_mean_3=np.load(path+'lse_soft_offline_beta_mean_item_10_d_3_t_9.npy')
beta_std_3=np.load(path+'lse_soft_offline_beta_std_item_10_d_3_t_9.npy')

beta_mean_5=np.load(path+'lse_soft_offline_beta_mean_item_10_d_5_t_9.npy')
beta_std_5=np.load(path+'lse_soft_offline_beta_std_item_10_d_5_t_9.npy')

beta_mean_10=np.load(path+'lse_soft_offline_beta_mean_item_20_d_10_t_9.npy')
beta_std_10=np.load(path+'lse_soft_offline_beta_std_item_20_d_10_t_9.npy')


# beta_mean_10=np.load(path+'lse_soft_offline_beta_mean_item_20_d_10_t_9.npy')
# beta_std_10=np.load(path+'lse_soft_offline_beta_std_item_20_d_10_t_9.npy')

# beta_mean_5=np.load(path+'lse_soft_offline_beta_mean_item_20_d_5_t_9.npy')
# beta_std_5=np.load(path+'lse_soft_offline_beta_std_item_20_d_5_t_9.npy')

# # beta_mean_15=np.load(path+'lse_soft_offline_beta_mean_item_20_d_15_t_9.npy')
# # beta_std_15=np.load(path+'lse_soft_offline_beta_std_item_20_d_15_t_9.npy')

x=range(train_loops)
plt.figure(figsize=(5,5))
# plt.plot(x, beta_mean_3,  '-.', color='c',markevery=0.1, markersize=5,linewidth=2, label='d = 3')
# plt.fill_between(x, beta_mean_3-beta_std_3*0.95, beta_mean_3+beta_std_3*0.95, color='c', alpha=0.2)
plt.plot(x, beta_mean_5, '-', color='y', markevery=0.1, markersize=5, linewidth=2, label='d=5')
plt.fill_between(x, beta_mean_5-beta_std_5*0.95, beta_mean_5+beta_std_5*0.95, color='y', alpha=0.2)
plt.plot(x, beta_mean_10, '-.', color='r', markevery=0.1, markersize=5, linewidth=2, label='d=7')
plt.fill_between(x, beta_mean_10-beta_std_10*0.95, beta_mean_10+beta_std_10*0.95, color='r', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'offline_beta_d'+'.png', dpi=300)
plt.show()



regret_mean_3=np.load(path+'lse_soft_offline_regret_mean_item_10_d_3_t_9.npy')
regret_std_3=np.load(path+'lse_soft_offline_regret_std_item_10_d_3_t_9.npy')

regret_mean_5=np.load(path+'lse_soft_offline_regret_mean_item_10_d_5_t_9.npy')
regret_std_5=np.load(path+'lse_soft_offline_regret_std_item_10_d_5_t_9.npy')

regret_mean_10=np.load(path+'lse_soft_offline_regret_mean_item_20_d_10_t_9.npy')
regret_std_10=np.load(path+'lse_soft_offline_regret_std_item_20_d_10_t_9.npy')

# regret_mean_7[76]=np.mean(regret_mean_7[[75,77]])
# regret_std_7[76]=np.mean(regret_std_7[[75,77]])

plt.figure(figsize=(5,5))
# plt.plot(x, regret_mean_3, '-.',color='c', markevery=0.1, markersize=5, linewidth=2, label='d = 3')
# plt.fill_between(x, regret_mean_3-regret_std_3*0.75, regret_mean_3+regret_std_3*0.75, color='c', alpha=0.2)
plt.plot(x, regret_mean_5,  '-',color='y', markevery=0.1, markersize=5,linewidth=2, label='d=5')
plt.fill_between(x, regret_mean_5-regret_std_5*0.95, regret_mean_5+regret_std_5*0.95, color='y', alpha=0.2)
plt.plot(x, regret_mean_10, '-.', color='r', markevery=0.1, markersize=5, linewidth=2, label='d=7')
plt.fill_between(x, regret_mean_10-regret_std_10*0.95, regret_mean_10+regret_std_10*0.95, color='r', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Cumulative regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'offline_regret_d'+'.png', dpi=300)
plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(x, regret_mean, color='c', linewidth=2)
# plt.fill_between(x, regret_mean-regret_std*0.95, regret_mean+regret_std*0.95, color='c', alpha=0.2)
# plt.xlabel('Training iteration', fontsize=12)
# plt.ylabel('Cumulative regret', fontsize=12)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_regret_list_train, color='c', linewidth=2)
# plt.ylabel('Cumulative Regret', fontsize=14)
# plt.xlabel('Training Iteration', fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_regret'+'.png', dpi=300)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_beta_list_train, color='c', linewidth=2)

# plt.ylabel('Beta', fontsize=14)
# plt.xlabel('Training Iteration', fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_beta'+'.png', dpi=300)
# plt.show()

# np.save(path+'lse_soft_offline_regret_%s_t_%s_d_%s.npy'%(item_num, phase_num, dimension), lse_soft_beta_list_train)
# np.save(path+'lse_soft_offline_beta_%s_t_%s_d_%s.npy'%(item_num, phase_num, dimension), lse_soft_beta_list_train)


# # test data
# linucb_regret_matrix=np.zeros((loop, iteration))
# lse_soft_regret_matrix=np.zeros((loop, iteration))
# for l in range(loop):

# 	item_features=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
# 	true_payoffs=np.dot(item_features, user_feature)
# 	best_arm=np.argmax(true_payoffs)
# 	worst_arm=np.argmin(true_payoffs)
# 	gaps=np.max(true_payoffs)-true_payoffs

# 	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, gaps)

# 	#####################

# 	linucb_regret, linucb_error, linucb_item_index, linucb_upper_matrix, linucb_low_matrix, linucb_payoff_error_matrix, linucb_worst_payoff_error, linucb_noise_norm, linucb_error_bound, linucb_threshold=linucb_model.run()

# 	lse_soft_regret, lse_soft_error, lse_soft_prob_matrix, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)


# 	linucb_regret_matrix[l]=linucb_regret
# 	lse_soft_regret_matrix[l]=lse_soft_regret


# linucb_mean=np.mean(linucb_regret_matrix, axis=0)
# linucb_std=linucb_regret_matrix.std(0)

# lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
# lse_soft_std=lse_soft_regret_matrix.std(0)



# x=range(iteration)
# plt.figure(figsize=(5,5))
# plt.plot(x, linucb_mean, '-.', color='b', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
# plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
# plt.plot(x, lse_soft_mean, '-p', color='c', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft')
# plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='c', alpha=0.2)
# plt.legend(loc=2, fontsize=12)
# # plt.ylim([0,300])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Cumulative Regret', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'regret_shadow_soft'+'.png', dpi=300)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(linucb_error, color='b', label='LinUCB')
# plt.plot(lse_soft_error, color='c', label='LSE-Soft')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_regret_list_train, color='b')
# plt.xlabel('Training iteration', fontsize=12)
# plt.ylabel('Cumulative Regret', fontsize=12)
# # plt.legend(loc=1, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_regret_d_%s'%(dimension)+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(lse_soft_beta_list_train, color='b')
# plt.xlabel('Training iteration', fontsize=12)
# plt.ylabel('Beta', fontsize=12)
# # plt.legend(loc=1, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_training_beta_d_%s'%(dimension)+'.png', dpi=200)
# plt.show()



# print(linucb_model.beta, lse_soft_model.beta, lse_soft_model.gamma)
# x=range(iteration)
# color_list=matplotlib.cm.get_cmap(name='Set3', lut=None).colors


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(x, lse_soft_prob_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
# 	else:
# 		plt.plot(x, lse_soft_prob_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
# plt.legend(loc=4, fontsize=12)
# plt.ylim([-0.15,1.1])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_prob_matrix'+'.png', dpi=300)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(x, np.zeros(iteration),'-', color='k', linewidth=2)
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(x, lse_soft_s_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
# 	else:
# 		plt.plot(x, lse_soft_s_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
# plt.legend(loc=4, fontsize=12)
# plt.ylim([-1,2])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('S(i,t)', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_s_matrix'+'.png', dpi=300)
# plt.show()

















