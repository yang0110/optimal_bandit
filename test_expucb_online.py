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
from expucb_online import Expucb_online
from utils import *
path='../results/lse_soft_results/'
np.random.seed(2018)

user_num=1
item_num=20
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.01
step_size_user_f=0.05
weight1=0.01
beta_online=2
gamma=0
time_width=20
loop_num=1

regret_matrix=np.zeros((loop_num, iteration))
beta_matrix=np.zeros((loop_num, iteration))
l_matrix=np.zeros((loop_num, iteration))
g_matrix=np.zeros((loop_num, iteration))
p_matrix=np.zeros((loop_num, iteration))
td_matrix=np.zeros((loop_num, iteration))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
item_features=np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.linalg.pinv(2*np.identity(dimension)), size=item_num)
item_features=Normalizer().fit_transform(item_features)
true_payoffs=np.dot(item_features, user_feature)
best_arm=np.argmax(true_payoffs)

for l in range(loop_num):
	# item_features=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
	online_model=Expucb_online(dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, step_size_beta,step_size_user_f, weight1, beta_online, gamma, time_width)
	online_regret, online_error, online_prob_matrix, online_s_matrix, online_beta_list, online_l_set, online_gradient, online_td=online_model.run()

	regret_matrix[l]=online_regret
	beta_matrix[l]=online_beta_list
	l_matrix[l]=online_l_set
	p_matrix[l]=online_prob_matrix[best_arm]
	g_matrix[l]=online_gradient
	td_matrix[l]=online_td

beta_mean=np.mean(beta_matrix, axis=0)
beta_std=beta_matrix.std(0)
regret_mean=np.mean(regret_matrix, axis=0)
regret_std=regret_matrix.std(0)
l_mean=np.mean(l_matrix, axis=0)
l_std=l_matrix.std(0)
p_mean=np.mean(p_matrix, axis=0)
p_std=p_matrix.std(0)
g_mean=np.mean(g_matrix, axis=0)
g_std=g_matrix.std(0)
td_mean=np.mean(td_matrix, axis=0)
td_std=td_mean.std(0)


np.save(path+'exp_online_beta_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), beta_mean)
np.save(path+'exp_online_beta_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), beta_std)

np.save(path+'exp_online_regret_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), regret_mean)
np.save(path+'exp_online_regret_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), regret_std)

np.save(path+'exp_online_l_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), l_mean)
np.save(path+'exp_online_l_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), l_std)


np.save(path+'exp_online_p_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), p_mean)
np.save(path+'exp_online_p_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), p_std)

np.save(path+'exp_online_g_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), g_mean)
np.save(path+'exp_online_g_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), g_std)

np.save(path+'exp_online_td_mean_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), td_mean)
np.save(path+'exp_online_td_std_item_%s_d_%s_t_%s_wind_%s_beta_%s.npy'%(item_num, dimension, phase_num, time_width,beta_online), td_std)

x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, beta_mean, color='y', linewidth=2, label='d = %s'%(dimension))
plt.fill_between(x, beta_mean-beta_std*0.95, beta_mean+beta_std*0.95, color='y', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'lse_soft_online_beta'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, regret_mean, color='r', linewidth=2, label='d = %s'%(dimension))
plt.fill_between(x, regret_mean-regret_std*0.95, regret_mean+regret_std*0.95, color='r', alpha=0.2)
plt.legend(loc=1, fontsize=12)
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'lse_soft_online_regret'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(online_prob_matrix[best_arm])
plt.xlabel('time', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.tight_layout()
plt.show()

x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, g_mean, '-', markevery=0.1, color='r')
plt.fill_between(x, g_mean-g_std*0.95, g_mean+g_std*0.95, color='r')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Gradient', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(td_mean)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Td-Error', fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
plt.plot(online_error)
plt.xlabel('Time', fontsize=14)
plt.ylabel('online error', fontsize=14)
plt.tight_layout()
plt.show()

# beta_mean_5=np.load(path+'online_beta_mean_item_20_d_5_t_11_wind_20.npy')
# beta_std_5=np.load(path+'online_beta_std_item_20_d_5_t_11_wind_20.npy')

# beta_mean_10=np.load(path+'online_beta_mean_item_20_d_10_t_11_wind_20.npy')
# beta_std_10=np.load(path+'online_beta_std_item_20_d_10_t_11_wind_20.npy')

# beta_mean_15=np.load(path+'online_beta_mean_item_20_d_15_t_11_wind_20.npy')
# beta_std_15=np.load(path+'online_beta_std_item_20_d_15_t_11_wind_20.npy')


# regret_mean_5=np.load(path+'online_regret_mean_item_20_d_5_t_11_wind_20.npy')
# regret_std_5=np.load(path+'online_regret_std_item_20_d_5_t_11_wind_20.npy')

# regret_mean_10=np.load(path+'online_regret_mean_item_20_d_10_t_11_wind_20.npy')
# regret_std_10=np.load(path+'online_regret_std_item_20_d_10_t_11_wind_20.npy')

# regret_mean_15=np.load(path+'online_regret_mean_item_20_d_15_t_11_wind_20.npy')
# regret_std_15=np.load(path+'online_regret_std_item_20_d_15_t_11_wind_20.npy')

# l_mean_5=np.load(path+'online_l_mean_item_20_d_5_t_11_wind_20.npy')
# l_std_5=np.load(path+'online_l_std_item_20_d_5_t_11_wind_20.npy')

# l_mean_10=np.load(path+'online_l_mean_item_20_d_10_t_11_wind_20.npy')
# l_std_10=np.load(path+'online_l_std_item_20_d_10_t_11_wind_20.npy')

# l_mean_15=np.load(path+'online_l_mean_item_20_d_15_t_11_wind_20.npy')
# l_std_15=np.load(path+'online_l_std_item_20_d_15_t_11_wind_20.npy')


# x=range(iteration)
# plt.figure(figsize=(5,5))
# plt.plot(x, beta_mean_5,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=5, T=2^11')
# plt.fill_between(x, beta_mean_5-beta_std_5*0.95, beta_mean_5+beta_std_5*0.95, color='c', alpha=0.2)
# plt.plot(x, beta_mean_10,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, T=2^11')
# plt.fill_between(x, beta_mean_10-beta_std_10*0.95, beta_mean_10+beta_std_10*0.95, color='y', alpha=0.2)
# plt.plot(x, beta_mean_15,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=15, T=2^11')
# plt.fill_between(x, beta_mean_15-beta_std_15*0.95, beta_mean_15+beta_std_15*0.95, color='r', alpha=0.2)
# # plt.ylim([1,5])
# plt.legend(loc=4, fontsize=12)
# plt.xlabel('Training iteration', fontsize=14)
# plt.ylabel('Beta', fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'online_beta_d'+'.png', dpi=100)
# plt.show()

# x=range(iteration)
# plt.figure(figsize=(5,5))
# plt.plot(x, regret_mean_5,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=5, T=2^11')
# plt.fill_between(x, regret_mean_5-regret_std_5*0.95, regret_mean_5+regret_std_5*0.95, color='c', alpha=0.2)
# plt.plot(x, regret_mean_10,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, T=2^11')
# plt.fill_between(x, regret_mean_10-regret_std_10*0.95, regret_mean_10+regret_std_10*0.95, color='y', alpha=0.2)
# plt.plot(x, regret_mean_15,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=15, T=2^11')
# plt.fill_between(x, regret_mean_15-regret_std_15*0.95, regret_mean_15+regret_std_15*0.95, color='r', alpha=0.2)
# # plt.ylim([1,5])
# plt.legend(loc=4, fontsize=12)
# plt.xlabel('Training iteration', fontsize=14)
# plt.ylabel('Cumulative Regret', fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'online_regret_d'+'.png', dpi=100)
# plt.show()




# beta_matrix=np.zeros((len(d_list), iteration))
# best_arm_prob_matrix=np.zeros((len(d_list), iteration))
# online_regret_matrix=np.zeros((len(d_list), iteration))

# for index, dimension in enumerate(d_list):
# 	user_feature=np.random.normal(size=dimension)
# 	user_feature=user_feature/np.linalg.norm(user_feature)
# 	item_features=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
# 	true_payoffs=np.dot(item_features, user_feature)
# 	best_arm=np.argmax(true_payoffs)
# 	worst_arm=np.argmin(true_payoffs)
# 	gaps=np.max(true_payoffs)-true_payoffs
# 	online_model=LSE_soft_online(dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, step_size_beta, weight1, beta_online, gamma, time_width)

# 	online_regret, online_error, online_prob_matrix, online_s_matrix, online_beta_list=online_model.run()
# 	online_regret_matrix[index]=online_regret
# 	beta_matrix[index]=online_beta_list
# 	best_arm_prob_matrix[index]=online_prob_matrix[best_arm]

# online_mean=np.mean(online_regret_matrix, axis=0)
# online_std=online_regret_matrix.std(0)


# x=range(iteration)
# plt.figure(figsize=(5,5))
# plt.plot(x, online_mean, '-o', color='k', markevery=0.1, linewidth=2, markersize=8, label='LSE-Soft-Online')
# plt.fill_between(x, online_mean-online_std*0.95, online_mean+online_std*0.95, color='k', alpha=0.2)
# plt.legend(loc=2, fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Cumulative Regret', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'regret_shadow_soft'+'.png', dpi=300)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(online_error, color='k', label='LSe-Soft-Online')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(len(d_list)):
# 	plt.plot(beta_matrix[i],linewidth=2, label='d = %s'%(d_list[i]))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Beta', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_online_training_beta_d'+'.png', dpi=200)
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(len(d_list)):
# 	plt.plot(online_regret_matrix[i],linewidth=2, label='%s'%(d_list[i]))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Regret', fontsize=12)
# plt.legend(loc=4, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_online_training_regret'+'.png', dpi=200)
# plt.show()

# plt.figure(figsize=(5,5))
# for i in range(len(d_list)):
# 	plt.plot(best_arm_prob_matrix[i], linewidth=2, label='%s'%(d_list[i]))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability of optimal arm', fontsize=12)
# plt.legend(loc=4, fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_online_training_beat_arm'+'.png', dpi=200)
# plt.show()

# x=range(iteration)
# color_list=matplotlib.cm.get_cmap(name='Set3', lut=None).colors


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(x, online_prob_matrix[i], '-', color='c', markevery=0.2, linewidth=2, markersize=8, label='Best Arm')
# 	else:
# 		plt.plot(x, online_prob_matrix[i], '-', color='gray', markevery=0.2, linewidth=2, markersize=5)
# plt.legend(loc=4, fontsize=12)
# plt.ylim([-0.15,1.1])
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Probability', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'lse_soft_online_prob_matrix'+'.png', dpi=300)
# plt.show()

















