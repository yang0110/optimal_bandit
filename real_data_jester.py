import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
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
input_path='../data/'
path='../results/real_data/'
# np.random.seed(2018)


jester=np.load(input_path+'jester.npy')

total_item_num=100
item_features=jester[:1000,:32].copy()
# item_features=Normalizer().fit_transform(item_features)
rewards=jester[:1000,35].copy()
# rewards=(rewards-np.min(rewards))/(np.max(rewards)-np.min(rewards))
mms=MinMaxScaler()
rewards=mms.fit_transform(rewards.reshape(-1,1))
# plt.hist(rewards)
# plt.show()
dim=10
pca=PCA(n_components=dim)
item_features=pca.fit_transform(item_features)
# item_features=Normalizer().fit_transform(item_features)

cov=0.1*np.identity(dim)
bias=np.zeros(dim)
for i in range(1000):
	x=item_features[i]
	cov+=np.outer(x,x)
	bias+=rewards[i]*x

user_feature=np.dot(np.linalg.pinv(cov), bias)
# user_feature=user_feature/np.linalg.norm(user_feature)
j_item=item_features.copy()
pred=np.dot(item_features, user_feature)
pred_error=np.linalg.norm(np.dot(item_features, user_feature)-rewards)
print('pred_error', pred_error)


user_num=1
item_num=20
dimension=10
phase_num=11
iteration=2**phase_num
sigma=0.1 # noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.01
step_size_gamma=0.01
weight1=0.01
loop=10

beta=1
gamma=0
train_loops=300

beta_online=0.5
time_width=20

epsilon=20
pseudo_num=1

linucb_regret_matrix=np.zeros((loop, iteration))
lints_regret_matrix=np.zeros((loop, iteration))
giro_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
online_regret_matrix=np.zeros((loop, iteration))
offline_prob_matrix=np.zeros((loop, iteration))
online_prob_matrix=np.zeros((loop, iteration))
online_beta_matrix=np.zeros((loop, iteration))



# train model
lse_soft_model=LSE_soft(dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

# lse_soft_regret_list_train, lse_soft_beta_list_train=lse_soft_model.train(train_loops, item_num)
random_item=np.random.choice(range(100), size=item_num)
item_features=j_item[random_item]
# item_features=Normalizer().fit_transform(item_features)
# true_payoffs=np.dot(item_features, user_feature)
true_payoffs=rewards[random_item]
best_arm=np.argmax(true_payoffs)

for l in range(loop):
	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, beta)
	lints_model=LINTS(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, epsilon)
	giro_model=GIRO(dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, pseudo_num)
	online_model=LSE_soft_online(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, sigma, step_size_beta, weight1, beta_online, gamma, time_width)

	#####################

	linucb_regret, linucb_error=linucb_model.run()
	lints_regret, lints_error=lints_model.run()
	giro_regret, giro_error=giro_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_prob_matrix, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)
	online_regret, online_error, online_soft_matrix, online_s_matrix, online_beta_list=online_model.run()

	linucb_regret_matrix[l]=linucb_regret
	lints_regret_matrix[l]=lints_regret
	giro_regret_matrix[l]=giro_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	online_regret_matrix[l]=online_regret
	offline_prob_matrix[l]=lse_soft_prob_matrix[best_arm]
	online_prob_matrix[l]=online_soft_matrix[best_arm]
	online_beta_matrix[l]=online_beta_list


linucb_mean=np.mean(linucb_regret_matrix, axis=0)
linucb_std=linucb_regret_matrix.std(0)

lints_mean=np.mean(lints_regret_matrix, axis=0)
lints_std=lints_regret_matrix.std(0)

lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)

online_mean=np.mean(online_regret_matrix, axis=0)
online_std=online_regret_matrix.std(0)

giro_mean=np.mean(giro_regret_matrix, axis=0)
giro_std=giro_regret_matrix.std(0)

offline_prob_mean=np.mean(offline_prob_matrix, axis=0)
offline_prob_std=offline_prob_matrix.std(0)

online_prob_mean=np.mean(online_prob_matrix, axis=0)
online_prob_std=online_prob_matrix.std(0)

online_beta_mean=np.mean(online_beta_matrix, axis=0)
online_beta_std=online_beta_matrix.std(0)

np.save(path+'jester_offline_prob_mean_d_%s'%(dimension), offline_prob_mean)
np.save(path+'jester_offline_prob_std_d_%s'%(dimension), offline_prob_std)
np.save(path+'jester_online_prob_mean_d_%s'%(dimension), online_prob_mean)
np.save(path+'jester_online_prob_std_d_%s'%(dimension), online_prob_std)

np.save(path+'jester_online_beta_mean_d_%s'%(dimension), online_beta_mean)
np.save(path+'jester_online_beta_std_d_%s'%(dimension), online_beta_std)

x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', color='b', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lints_mean, '-', color='g', markevery=0.1, linewidth=2, markersize=8, label='LinTS')
plt.fill_between(x, lints_mean-lints_std*0.95, lints_mean+lints_std*0.95, color='g', alpha=0.2)

# plt.plot(x, giro_mean, '-*', color='c', markevery=0.1, linewidth=2, markersize=8, label='Giro')
# plt.fill_between(x, giro_mean-giro_std*0.5, giro_mean+giro_std*0.5, color='c', alpha=0.2)

plt.plot(x, lse_soft_mean, '-p', color='r', markevery=0.1, linewidth=2, markersize=8, label='ExpUCB (offline)')
plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='r', alpha=0.2)
plt.plot(x, online_mean, '-o', color='orange', markevery=0.1, linewidth=2, markersize=8, label='ExpUCB (online)')
plt.fill_between(x, online_mean-online_std*0.95, online_mean+online_std*0.95, color='orange', alpha=0.2)
plt.legend(loc=2, fontsize=12)
# plt.ylim([0,250])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'jester_regret_shadow_soft_d_%s'%(dimension)+'.png', dpi=300)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(linucb_error, color='b', label='LinUCB')
plt.plot(lints_error,color='g', label='LinTS')
plt.plot(giro_error, color='c', label='Giro')
plt.plot(lse_soft_error, color='r', label='ExpUCB (offline)')
plt.plot(online_error, color='orange', label='ExpUCB (online)')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'jester_error_shadow_soft_d_%s'%(dimension)+'.png', dpi=300)
plt.show()

# offline_prob_mean_5=np.load(path+'offline_prob_mean_d_5.npy')
# offline_prob_std_5=np.load(path+'offline_prob_std_d_5.npy')
# offline_prob_mean_10=np.load(path+'offline_prob_mean_d_10.npy')
# offline_prob_std_10=np.load(path+'offline_prob_std_d_10.npy')


# online_prob_mean_5=np.load(path+'online_prob_mean_d_5.npy')
# online_prob_std_5=np.load(path+'online_prob_std_d_5.npy')
# online_prob_mean_10=np.load(path+'online_prob_mean_d_10.npy')
# online_prob_std_10=np.load(path+'online_prob_std_d_10.npy')


plt.figure(figsize=(5,5))
plt.plot(x, offline_prob_mean, '-o', color='r', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (offline)')
plt.fill_between(x, offline_prob_mean-offline_prob_std*0.95, offline_prob_mean+offline_prob_std*0.95, color='r', alpha=0.2)
plt.plot(x, online_prob_mean, '-*', color='orange', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (online)')
plt.fill_between(x, online_prob_mean-online_prob_std*0.95, online_prob_mean+online_prob_std*0.95, color='orange', alpha=0.2)
plt.legend(loc=4, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Probability of optimal arm', fontsize=14)
plt.tight_layout()
plt.savefig(path+'jester_online_offline_prob'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, online_beta_mean, '-*', color='orange', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (online)')
plt.fill_between(x, online_beta_mean-online_beta_std*0.95, online_beta_mean+online_beta_std*0.95, color='orange', alpha=0.2)
plt.legend(loc=4, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'jester_online_beta'+'.png', dpi=300)
plt.show()


movielens_offline_prob_mean=np.load(path+'movielens_offline_prob_mean_d_5.npy')
movielens_offline_prob_std=np.load(path+'movielens_offline_prob_std_d_5.npy')
jester_offline_prob_mean=np.load(path+'jester_offline_prob_mean_d_10.npy')
jester_offline_prob_std=np.load(path+'jester_offline_prob_std_d_10.npy')


movielens_online_prob_mean=np.load(path+'movielens_online_prob_mean_d_5.npy')
movielens_online_prob_std=np.load(path+'movielens_online_prob_std_d_5.npy')
jester_online_prob_mean=np.load(path+'jester_online_prob_mean_d_10.npy')
jester_online_prob_std=np.load(path+'jester_online_prob_std_d_10.npy')


x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, movielens_offline_prob_mean, '-o', color='r', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (offline)-Movielens')
plt.fill_between(x, movielens_offline_prob_mean-movielens_offline_prob_std*0.95, movielens_offline_prob_mean+movielens_offline_prob_std*0.95, color='r', alpha=0.2)
plt.plot(x, movielens_online_prob_mean, '-*', color='orange', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (online)-Movielens')
plt.fill_between(x, movielens_online_prob_mean-movielens_online_prob_std*0.95, movielens_online_prob_mean+movielens_online_prob_std*0.95, color='orange', alpha=0.2)
plt.plot(x, jester_offline_prob_mean, '-s', color='pink', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (offline)-Jester')
plt.fill_between(x, jester_offline_prob_mean-jester_offline_prob_std*0.95, jester_offline_prob_mean+jester_offline_prob_std*0.95, color='pink', alpha=0.2)
plt.plot(x, jester_online_prob_mean, '-|', color='y', linewidth=2, markevery=0.1, markersize=5, label='ExpUCB (online)-Jester')
plt.fill_between(x, jester_online_prob_mean-jester_online_prob_std*0.95, jester_online_prob_mean+jester_online_prob_std*0.95, color='y', alpha=0.2)
plt.legend(loc=4, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Probability of optimal arm', fontsize=14)
plt.tight_layout()
plt.savefig(path+'movielens_jester_online_offline_prob'+'.png', dpi=300)
plt.show()




# online_beta_mean_5=np.load(path+'online_beta_mean_d_5.npy')
# online_beta_std_5=np.load(path+'online_beta_std_d_5.npy')
# online_beta_mean_10=np.load(path+'online_beta_mean_d_10.npy')
# online_beta_std_10=np.load(path+'online_beta_std_d_10.npy')

# plt.figure(figsize=(5,5))
# plt.plot(x, online_beta_mean_5, '-p',  color='orange', linewidth=2, markevery=0.2, markersize=5, label='ExpUCB (onlline), d = 5')
# plt.fill_between(x, online_beta_mean_5-online_beta_std_5*0.95, online_beta_mean_5+online_beta_std_5*0.95, color='orange', alpha=0.2)
# plt.plot(x, online_beta_mean_10, '-o',  color='y', linewidth=2, markevery=0.2, markersize=5, label='ExpUCB (onlline), d = 10')
# plt.fill_between(x, online_beta_mean_10-online_beta_std_10*0.95, online_beta_mean_10+online_beta_std_10*0.95, color='y', alpha=0.2)
# plt.ylim([0,5])
# plt.legend(loc=4, fontsize=12)
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Beta', fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'simu_online_beta'+'.png', dpi=300)
# plt.show()



