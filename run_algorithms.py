## Fix, finite arm set, the set set in each round
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
# os.chdir('C:/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from linucb_eli import LINUCB_ELI
from eliminator import ELI
from lse import LSE 
from se import SE
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=10
dimension=5
phase_num=10
iteration=2**phase_num
sigma=0.1# noise
delta=0.1# high probability
alpha=1 # regularizer
state=1 # small beta (exploitation), large beta(exploration), 1: true beta
combine_method=1
lambda_=1

item_feature=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
true_payoffs=np.dot(item_feature, user_feature)
best_arm=np.argmax(true_payoffs)


linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, state)

linucb_eli_model=LINUCB_ELI(dimension, iteration, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma, state)

eli_model=ELI(dimension, phase_num, item_num, user_feature,item_feature, true_payoffs, alpha, delta, sigma)

se_model=SE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma)

lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, lambda_, combine_method)
#####################

linucb_regret, linucb_error, linucb_item_index, linucb_x_norm_matrix=linucb_model.run(iteration)

linucb_eli_regret, linucb_eli_error, linucb_eli_item_index, linucb_eli_x_norm_matrix=linucb_eli_model.run(iteration)

eli_regret, eli_error, eli_item_index, eli_x_norm_matrix=eli_model.run(iteration)

se_regret, se_error, se_item_index, se_x_norm_matrix=se_model.run(iteration)

lse_regret, lse_error, lse_item_index, lse_x_norm_matrix, lse_avg_x_norm_matrix=lse_model.run(iteration)

plt.figure(figsize=(5,5))
plt.plot(linucb_regret, label='LinUCB')
plt.plot(linucb_eli_regret, label='LinUCB ELI')
plt.plot(eli_regret, label='Eliminator')
plt.plot(se_regret, label='SE')
plt.plot(lse_regret, label='LSE')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LinUCB')
plt.plot(linucb_eli_error, label='LinUCB ELI')
plt.plot(eli_error, label='Eliminator')
plt.plot(se_error, label='SE')
plt.plot(lse_error, label='LSE')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.show()


fig, ax=plt.subplots(1,2)
for i in range(item_num):
	ax[0].plot(linucb_x_norm_matrix[i], label='arm=%s'%(i))
	ax[1].plot(linucb_eli_x_norm_matrix[i], label='arm=%s'%(i))
ax[0].set_title('LinUCB (Best arm=%s)'%(best_arm))
ax[1].set_title('LinUCB ELI (Best arm=%s)'%(best_arm))
ax[0].set_ylabel('x_norm')
ax[0].legend(loc=1)
ax[1].legend(loc=1)
plt.show()


fig, ax=plt.subplots(1,2)
for i in range(item_num):
	ax[0].plot(eli_x_norm_matrix[i], label='arm=%s'%(i))
	ax[1].plot(se_x_norm_matrix[i], label='arm=%s'%(i))
ax[0].set_title('ELI (Best arm=%s)'%(best_arm))
ax[1].set_title('SE (Best arm=%s)'%(best_arm))
ax[0].set_ylabel('x_norm')
ax[0].legend(loc=1)
ax[1].legend(loc=1)
plt.show()

fig, ax=plt.subplots(1,2)
for i in range(item_num):
	ax[0].plot(se_x_norm_matrix[i], label='arm=%s'%(i))
	ax[1].plot(lse_x_norm_matrix[i], label='arm=%s'%(i))
ax[0].set_title('SE (Best arm=%s)'%(best_arm))
ax[1].set_title('LSE (current) (Best arm=%s)'%(best_arm))
ax[0].set_ylabel('x_norm')
ax[0].legend(loc=1)
ax[1].legend(loc=1)
plt.show()



fig, ax=plt.subplots(1,2)
for i in range(item_num):
	ax[0].plot(eli_x_norm_matrix[i], label='arm=%s'%(i))
	ax[1].plot(lse_x_norm_matrix[i], label='arm=%s'%(i))
ax[0].set_title('ELI (Best arm=%s)'%(best_arm))
ax[1].set_title('LSE (current) (Best arm=%s)'%(best_arm))
ax[0].set_ylabel('x_norm')
ax[0].legend(loc=1)
ax[1].legend(loc=1)
plt.show()

fig, ax=plt.subplots(1,2)
for i in range(item_num):
	ax[0].plot(lse_x_norm_matrix[i], label='arm=%s'%(i))
	ax[1].plot(lse_avg_x_norm_matrix[i], label='arm=%s'%(i))
ax[0].set_title('LSE (current) (Best arm=%s)'%(best_arm))
ax[1].set_title('LSE (combined) (Best arm=%s)'%(best_arm))
ax[0].set_ylabel('x_norm')
ax[0].legend(loc=1)
ax[1].legend(loc=1)
plt.show()

# fig, ax=plt.subplots(1,2)
# ax[0].plot(linucb_item_index, '.')
# ax[1].plot(linucb_eli_item_index, '.')
# ax[0].set_title('LinUCB (Best arm=%s)'%(best_arm))
# ax[1].set_title('LinUCB ELI (Best arm=%s)'%(best_arm))
# ax[0].set_ylabel('Selected Item')
# plt.show()



