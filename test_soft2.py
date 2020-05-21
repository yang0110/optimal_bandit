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
# np.random.seed(2018)

user_num=1
item_num=50
dimension=5
phase_num=8
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1# regularizer
step_size_beta=0.01
step_size_gamma=0.02
weight1=0.01
train_loops=100
beta=5
gamma=0
loop_num=10
 
beta_list=list(np.linspace(0.01, 0.5, 50))+list(np.linspace(0.51,2,50))
user_feature=np.random.normal(size=dimension)
# user_feature=np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.identity(dimension))
# item_features=np.random.uniform(low=-1, high=1, size=(item_num, dimension))
user_feature=user_feature/np.linalg.norm(user_feature)
item_features=np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.linalg.pinv(np.identity(dimension)), size=item_num)
item_features=Normalizer().fit_transform(item_features)
true_payoffs=np.dot(item_features, user_feature)
best_arm=np.argmax(true_payoffs)

beta_matrix=np.zeros((loop_num, train_loops))
regret_matrix=np.zeros((loop_num, train_loops))
regret_list=np.zeros(train_loops)
for l in range(loop_num):
	print('loop', l)
	for index, beta in enumerate(beta_list):
		print('test_iteration', index)
		lse_soft_model=LSE_soft(dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)
		# lse_soft_regret_list_train, lse_soft_beta_list_train, lse_soft_prob_matrix, lse_soft_beta_gradient=lse_soft_model.train(train_loops, item_num)

		lse_soft_regret, lse_soft_error, lse_soft_prob_matrix, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)
		# beta_matrix[l]=lse_soft_beta_list_train
		# regret_matrix[l]=lse_soft_regret_list_train

		regret_list[index]=lse_soft_regret[-1]

	regret_matrix[l]=regret_list


regret_mean=np.mean(regret_matrix, axis=0)
regret_std=regret_matrix.std(0)


np.save(path+'regret_mean_vs_beta_d_%s_item_num_%s_t_%s'%(dimension, item_num, phase_num), regret_mean)

np.save(path+'regret_std_vs_beta_d_%s_item_num_%s_t_%s'%(dimension, item_num, phase_num), regret_std)

# x=np.range(train_loops)
plt.figure(figsize=(5,5))
plt.plot(beta_list, regret_mean, color='r')
# plt.fill_between(beta_list, regret_mean_5-regret_std_5*0.95, regret_mean_5+regret_std_5*0.95, color='r', alpha=0.2)
plt.ylabel('regret')
plt.xlabel('beta')
plt.savefig(path+'regret_va_beta_d_%s_item_num_%s_t_%s'%(dimension, item_num, phase_num)+'.png', dpi=300)
plt.show()




# plt.plot(beta_list, regret_list)
# plt.xlabel('beta', fontsize=12)
# plt.ylabel('regret')
# plt.show()


# plt.plot(lse_soft_regret)
# plt.ylabel('regret')
# plt.show()

# plt.plot(lse_soft_error)
# plt.ylabel('error')
# plt.show()



# np.save(path+'regret_mean_vs_beta_d_%s_item_num_%s_t_%s'%(dimension, item_num, phase_num), regret_mean)

# np.save(path+'regret_std_vs_beta_d_%s_item_num_%s_t_%s'%(dimension, item_num, phase_num), regret_std)

regret_mean_5_8=np.load(path+'regret_mean_vs_beta_d_5_item_num_20_t_8.npy')
regret_std_5_8=np.load(path+'regret_std_vs_beta_d_5_item_num_20_t_8.npy')

regret_mean_5_9=np.load(path+'regret_mean_vs_beta_d_5_item_num_20_t_9.npy')
regret_std_5_9=np.load(path+'regret_std_vs_beta_d_5_item_num_20_t_9.npy')

regret_mean_5_10=np.load(path+'regret_mean_vs_beta_d_5_item_num_20_t_10.npy')
regret_std_5_10=np.load(path+'regret_std_vs_beta_d_5_item_num_20_t_10.npy')

# regret_mean_5_11=np.load(path+'regret_mean_vs_beta_d_5_item_num_20_t_11.npy')
# regret_std_5_11=np.load(path+'regret_std_vs_beta_d_5_item_num_20_t_11.npy')

# regret_mean_7=np.load(path+'regret_mean_vs_beta_d_7_item_num_10_t_9.npy')
# regret_std_7=np.load(path+'regret_std_vs_beta_d_5_item_num_10_t_8.npy')

regret_mean_10=np.load(path+'regret_mean_vs_beta_d_10_item_num_20_t_10.npy')
regret_std_10=np.load(path+'regret_std_vs_beta_d_10_item_num_20_t_10.npy')
regret_mean_15=np.load(path+'regret_mean_vs_beta_d_15_item_num_20_t_10.npy')
regret_std_15=np.load(path+'regret_std_vs_beta_d_15_item_num_20_t_10.npy')

# plt.figure(figsize=(5,5))
# plt.plot(range(len(regret_mean_5)), regret_mean_5, color='r')
# plt.fill_between(range(len(regret_mean_5)), regret_mean_5-regret_std_5*0.95, regret_mean_5+regret_std_5*0.95, color='r', alpha=0.2)
# plt.ylabel('regret')
# plt.xlabel('beta')
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(range(len(regret_mean_7)), regret_mean_7, color='r')
# plt.fill_between(range(len(regret_mean_7)), regret_mean_7-regret_std_7*0.95, regret_mean_7+regret_std_7*0.95, color='r', alpha=0.2)
# plt.ylabel('regret')
# plt.xlabel('beta')
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(range(len(regret_mean_10)), regret_mean_10, color='r')
# plt.fill_between(range(len(regret_mean_10)), regret_mean_10-regret_std_10*0.95, regret_mean_10+regret_std_10*0.95, color='r', alpha=0.2)
# plt.ylabel('regret')
# plt.xlabel('beta')
# plt.show()

def avg_mean(data):
	new_data=np.zeros(len(data)-10)
	for i in range(len(data)-10):
		new_data[i]=np.mean(data[i:i+10])
	# for j in range(len(data)-10, len(data)):
	# 	new_data[j]=data[j]
	return new_data


plt.figure(figsize=(5,5))
plt.plot(beta_list[:-10], avg_mean(regret_mean_5_8), color='g', label='d=5, T=8')
# plt.plot(beta_list[:-10], avg_mean(regret_mean_5_9), color='c', label='d=5, T=9')
plt.plot(beta_list[:-10], avg_mean(regret_mean_5_10), color='b', label='d=5, T=10')
plt.plot(beta_list[:-10], avg_mean(regret_mean_10), color='r', label='d=10, T=10')
plt.plot(beta_list[:-10], avg_mean(regret_mean_15), color='y', label='d=15, T=10')
plt.legend(loc=1, fontsize=12)
plt.ylabel('Regret', fontsize=14)
plt.xlabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'avg_mean_regret_vs_beta_d_5_10_15'+'.png', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(beta_list, avg_mean(regret_mean_5_8), color='g', label='d=5, T=8')
plt.fill_between(beta_list, avg_mean(regret_mean_5_8)-avg_mean(regret_std_5_8)*0.95, avg_mean(regret_mean_5_8)+avg_mean(regret_std_5_8)*0.95, color='g', alpha=0.1)

plt.plot(beta_list, avg_mean(regret_mean_5_9), color='b', label='d=5, T=9')
plt.fill_between(beta_list, avg_mean(regret_mean_5_9)-avg_mean(regret_std_5_9)*0.95, avg_mean(regret_mean_5_9)+avg_mean(regret_std_5_9)*0.95, color='g', alpha=0.1)

plt.plot(beta_list, avg_mean(regret_mean_5_10), color='c', label='d=5, T=10')
plt.fill_between(beta_list, avg_mean(regret_mean_5_10)-avg_mean(regret_std_5_10)*0.95, avg_mean(regret_mean_5_10)+avg_mean(regret_std_5_10)*0.95, color='g', alpha=0.1)


# plt.plot(beta_list, regret_mean_5_11, color='c', label='d=5, T=10')
# plt.fill_between(beta_list, regret_mean_5_11-regret_std_5_11*0.95, regret_mean_5_11+regret_std_5_11*0.95, color='c', alpha=0.1)

plt.plot(beta_list, avg_mean(regret_mean_10), color='y', label='d=10, T=10')
plt.fill_between(beta_list, avg_mean(regret_mean_10)-avg_mean(regret_std_10)*0.95, avg_mean(regret_mean_10)+avg_mean(regret_std_10)*0.95, color='y', alpha=0.1)

plt.plot(beta_list, avg_mean(regret_mean_15), color='r', label='d=15, T=10')
plt.fill_between(beta_list, avg_mean(regret_mean_15)-avg_mean(regret_std_15)*0.95, avg_mean(regret_mean_15)+avg_mean(regret_std_15)*0.95, color='r', alpha=0.1)


plt.legend(loc=1, fontsize=12)
plt.ylabel('Regret', fontsize=14)
plt.xlabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'avg_mean_and_variance_regret_vs_beta_d_5_10_15'+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(beta_list, regret_mean_5_8, color='g', label='d=5, T=8')
plt.fill_between(beta_list, regret_mean_5_8-regret_std_5_8*0.95, regret_mean_5_8+regret_std_5_8*0.95, color='g', alpha=0.1)

# plt.plot(beta_list, regret_mean_5_9, color='b', label='d=5, T=9')
# plt.fill_between(beta_list, regret_mean_5_9-regret_std_5_9*0.95, regret_mean_5_9+regret_std_5_9*0.95, color='b', alpha=0.1)


plt.plot(beta_list, regret_mean_5_10, color='c', label='d=5, T=10')
plt.fill_between(beta_list, regret_mean_5_10-regret_std_5_10*0.95, regret_mean_5_10+regret_std_5_10*0.95, color='c', alpha=0.1)

plt.plot(beta_list, regret_mean_10, color='r', label='d=10, T=10')
plt.fill_between(beta_list, regret_mean_10-regret_std_10*0.95, regret_mean_10+regret_std_10*0.95, color='r', alpha=0.1)

plt.plot(beta_list, regret_mean_15, color='y', label='d=15, T=10')
plt.fill_between(beta_list, regret_mean_15-regret_std_15*0.95, regret_mean_15+regret_std_15*0.95, color='y', alpha=0.1)

# plt.plot(beta_list, regret_mean_5_11, color='c', label='d=5, T=10')
# plt.fill_between(beta_list, regret_mean_5_11-regret_std_5_11*0.95, regret_mean_5_11+regret_std_5_11*0.95, color='c', alpha=0.1)


plt.legend(loc=1, fontsize=12)
plt.ylabel('Regret', fontsize=14)
plt.xlabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'original_regret_vs_beta_d_5_10_15'+'.png', dpi=300)
plt.show()




