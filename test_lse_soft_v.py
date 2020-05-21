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
phase_num=9
iteration=2**phase_num
sigma=0.001# noise
delta=0.1# high probability
alpha=0.1 # regularizer
step_size_beta=0.001
step_size_gamma=0.05
weight1=0.01
test_loops=10
train_loops=100

# # training phase
# beta_list=[1,2,3,4]
# gamma_list=[1,3,5,7]
beta=5
gamma=0

# train_regret_matrix=np.zeros((len(beta_list), train_loops))
# train_gamma_matrix=np.zeros((len(beta_list), train_loops))
# train_beta_matrix=np.zeros((len(beta_list), train_loops))

# for index, beta in enumerate(beta_list):

lse_soft_v_model=LSE_soft_v(dimension, iteration, item_num, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

lse_soft_v_regret_list_train, lse_soft_v_beta_list_train=lse_soft_v_model.train(train_loops, item_num)

# train_regret_matrix[index]=lse_soft_v_regret_list_train
# train_beta_matrix[index]=lse_soft_v_beta_list_train


# testing phase
lse_soft_v_regret_matrix=np.zeros((test_loops, iteration))
lse_soft_v_max_matrix=np.zeros((test_loops, iteration))


item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)
worst_arm=np.argmin(true_payoffs)
gaps=np.max(true_payoffs)-true_payoffs

for l in range(test_loops):
	lse_soft_v_regret, lse_soft_v_error, lse_soft_v_max=lse_soft_v_model.run(user_feature, item_feature, true_payoffs)

	lse_soft_v_regret_matrix[l]=lse_soft_v_regret
	lse_soft_v_max_matrix[l]=lse_soft_v_max[best_arm]

lse_soft_v_mean=np.mean(lse_soft_v_regret_matrix, axis=0)
lse_soft_v_std=lse_soft_v_regret_matrix.std(0)

lse_soft_v_max_mean=np.mean(lse_soft_v_max_matrix, axis=0)
lse_soft_v_max_std=lse_soft_v_max_matrix.std(0)


x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, lse_soft_v_mean, '-.', color='c', markevery=0.1, linewidth=2, markersize=8)
plt.fill_between(x, lse_soft_v_mean-lse_soft_v_std*0.95, lse_soft_v_mean+lse_soft_v_std*0.95, color='c', alpha=0.2)
# plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_v_regret_shadow'+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(lse_soft_v_regret_list_train)
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Training cumulative regret', fontsize=12)
# plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_v_training_regret'+'.png', dpi=200)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(lse_soft_v_beta_list_train)
plt.xlabel('Training iteration', fontsize=12)
plt.ylabel('Learned Beta', fontsize=12)
# plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_v_training_beta'+'.png', dpi=200)
plt.show()


x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab10', lut=None).colors


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x, lse_soft_v_max[i], '-*', color='m', markevery=0.1, linewidth=2, markersize=5, label='Best Arm')
	else:
		plt.plot(x, lse_soft_v_max[i], '-.', color=color_list[i], markevery=0.2, linewidth=2, markersize=5)
plt.legend(loc=4, fontsize=12)
plt.ylim([-0.15,1.1])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
# plt.title('Best Arm=%s'%(best_arm), fontsize=12)
plt.tight_layout()
plt.savefig(path+'lse_soft_v_prob_matrix'+'.png', dpi=300)
plt.show()










