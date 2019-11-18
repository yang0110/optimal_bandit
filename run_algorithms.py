## Fix, finite arm set, the set set in each round
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from utils import *
path='../results/'
np.random.seed(2018)

user_num=1
item_num=10
dimension=5
pool_size=item_num
iteration=1000
sigma=0.1# noise
delta=0.1# high probability
alpha=1 # regularizer

item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
norm=np.linalg.norm(user_feature)
user_feature=user_feature/norm
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
true_payoffs=np.dot(item_feature_matrix, user_feature)+noise_matrix
true_payoffs=true_payoffs.ravel()
best_arm=np.argmax(np.dot(item_feature_matrix, user_feature))
best_payoff=np.max(np.dot(item_feature_matrix, user_feature))
gaps=best_payoff-np.dot(item_feature_matrix, user_feature)

linucb_model=LINUCB(dimension, iteration, item_num, item_feature_matrix, user_feature, true_payoffs, gaps, best_arm, best_payoff, alpha, delta, sigma)
regret, error, beta_list, true_beta_list, item_index, index_matrix, x_norm_matrix, mean_matrix, gaps_ucb, est_gaps_ucb, best_index, ucb_matrix=linucb_model.run(iteration)


plt.figure(figsize=(5,5))
plt.plot(item_index, '.', label='Best arm=%s'%(best_arm))
plt.xlabel('Time', fontsize=12)
plt.ylabel('Arm index', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		plt.plot(gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
	else:
		plt.plot(gaps_ucb[:,i], label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('Gap-UCB', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

fig, ax = plt.subplots(1,4)
ax[0].plot(item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		ax[1].plot(gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[2].plot(best_index[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[3].plot(est_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[1].plot(gaps_ucb[:,i], label='arm= %s'%(i))
		ax[2].plot(best_index[:,i], label='arm= %s'%(i))
		ax[3].plot(est_gaps_ucb[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[3].legend(loc=4)
ax[0].set_ylabel('arm selected')
ax[1].set_ylabel('gaps-ucb-ucb')
ax[2].set_ylabel('best-index')
ax[3].set_ylabel('est_gap-ucb-ucb')
plt.show()



plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
	else:
		plt.plot(index_matrix[:,i], label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('UCB of arm', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(mean_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
	else:
		plt.plot(mean_matrix[:,i], label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('est mean of arm', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(x_norm_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
	else:
		plt.plot(x_norm_matrix[:,i], label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('x norm of arm', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(beta_list, label='beta')
plt.plot(true_beta_list, label='true beta')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Beta', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(ucb_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
	else:
		plt.plot(ucb_matrix[:,i], label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('ucb matrix', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(gaps, label='Gaps')
plt.xlabel('Arm index', fontsize=12)
plt.ylabel('Gap', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(regret, label='Regret')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Regret', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(error, label='error')
plt.xlabel('Time', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.show()

