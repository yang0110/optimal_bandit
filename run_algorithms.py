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
from linucb_test import LINUCB_TEST
from oful import OFUL
from ucb1 import UCB1
from incremental_eliminator import INC_ELI
from utils import *
path='../results/'
np.random.seed(2018)

user_num=1
item_num=10
dimension=5
pool_size=item_num
iteration=3000
sigma=0.1# noise
delta=0.1# high probability
alpha=1 # regularizer
state=1 # small beta (exploitation), large beta(exploration), 1: true beta

item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
user_feature=np.random.normal(size=dimension)
norm=np.linalg.norm(user_feature)
user_feature=user_feature/norm
noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
true_payoffs=np.dot(item_feature_matrix, user_feature)+noise_matrix
true_payoffs=true_payoffs.ravel()
best_arm=np.argmax(np.dot(item_feature_matrix, user_feature))
worse_arm=np.argmin(np.dot(item_feature_matrix, user_feature))
best_payoff=np.max(np.dot(item_feature_matrix, user_feature))
gaps=best_payoff-np.dot(item_feature_matrix, user_feature)

linucb_model=LINUCB(dimension, iteration, item_num, item_feature_matrix, user_feature, true_payoffs, gaps, best_arm, best_payoff, alpha, delta, sigma, state)

eli_model=INC_ELI(dimension, iteration, user_feature, item_num, item_feature_matrix, true_payoffs, alpha, sigma, delta)

linucb_test_model=LINUCB_TEST(dimension, iteration, item_num, item_feature_matrix, user_feature, true_payoffs, gaps, best_arm, best_payoff, alpha, delta, sigma, state)

oful_model=OFUL(dimension, iteration, item_num, item_feature_matrix, user_feature, true_payoffs, gaps, best_arm, best_payoff, alpha, delta, sigma, state)

ucb1_model=UCB1(dimension, iteration, item_num, item_feature_matrix, user_feature, true_payoffs, gaps, best_arm, best_payoff, alpha, delta, sigma, state)

#####################

regret, error, beta_list, true_beta_list, item_index, index_matrix, x_norm_matrix, mean_matrix, gaps_ucb, est_gaps_ucb, best_index, ucb_matrix, payoff_error_matrix, ucb_list, true_ucb_list=linucb_model.run(iteration)

eli_regret, eli_error, eli_item_index, eli_remain_item, eli_beta_list, eli_x_norm_matrix, eli_index_matrix, eli_ucb_matrix, eli_mean_matrix, eli_payoff_error_matrix, eli_ucb_list, eli_true_ucb_list=eli_model.run()

linucb_test_regret, linucb_test_error, linucb_test_beta_list, linucb_test_true_beta_list, linucb_test_item_index, linucb_test_index_matrix, linucb_test_x_norm_matrix, linucb_test_mean_matrix, linucb_test_gaps_ucb, linucb_test_est_gaps_ucb, linucb_test_best_index, linucb_test_ucb_matrix, linucb_test_payoff_error_matrix=linucb_test_model.run(iteration)

oful_regret, oful_error, oful_beta_list, oful_true_beta_list, oful_item_index, oful_index_matrix, oful_x_norm_matrix, oful_mean_matrix, oful_gaps_ucb, oful_est_gaps_ucb, oful_best_index, oful_ucb_matrix, oful_payoff_error_matrix=oful_model.run(iteration)

ucb1_regret, ucb1_error, ucb1_beta_list, ucb1_true_beta_list, ucb1_item_index, ucb1_index_matrix, ucb1_x_norm_matrix, ucb1_mean_matrix, ucb1_gaps_ucb, ucb1_est_gaps_ucb, ucb1_best_index, ucb1_ucb_matrix, ucb1_payoff_error_matrix=ucb1_model.run(iteration)

def bounds(iteration, dimension, item_num, delta):
	bound1=np.zeros(iteration)
	bound2=np.zeros(iteration)
	for t in range(iteration):
		bound1[t]=dimension*np.sqrt(t)*np.log(t)
		bound2[t]=np.sqrt(t*dimension*np.log(item_num*t))
	return bound1, bound2 

b1, b2=bounds(100, dimension, item_num*100, delta)

plt.figure(figsize=(5,5))
plt.plot(b1, label='LinUCB bound')
plt.plot(b2, label='ELI bound')
plt.xlabel('time', fontsize=12)
plt.ylabel('Bound', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(gaps, '.-', color='r', label='Gaps')
plt.xlabel('Arm index', fontsize=12)
plt.ylabel('Gap', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(eli_remain_item, label='item num')
plt.xlabel('Phase', fontsize=12)
plt.ylabel('Remained item num', fontsize=12)
plt.title('ELI', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(ucb_list, label='LinUCB')
plt.plot(true_ucb_list, label='LinUCB True')
plt.plot(eli_ucb_list, label='ELI')
plt.plot(eli_true_ucb_list, label='ELI True')
plt.xlabel('Time', fontsize=12)
plt.ylabel('UCB', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(eli_x_norm_matrix[:,i], '*-', markevery=0.01, color='k', label='Best arm= %s'%(i))
	else:
		plt.plot(eli_x_norm_matrix[:,i], '.-', markevery=0.01, label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('x norm ', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.title('ELI', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(eli_index_matrix[:,i], '*-', markevery=0.01, color='k', label='Best arm= %s'%(i))
	else:
		plt.plot(eli_index_matrix[:,i], '.-', markevery=0.01, label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('index ', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.title('ELI', fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(eli_ucb_matrix[:,i], '*-', markevery=0.01, color='k', label='Best arm= %s'%(i))
	else:
		plt.plot(eli_ucb_matrix[:,i], '.-', markevery=0.01, label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('ucb ', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.title('ELI', fontsize=12)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	if i==best_arm:
		plt.plot(eli_payoff_error_matrix[:,i], '*-', markevery=0.01, color='k', label='Best arm= %s'%(i))
	else:
		plt.plot(eli_payoff_error_matrix[:,i], '.-', markevery=0.01, label='arm= %s'%(i))
plt.xlabel('Time', fontsize=12)
plt.ylabel('payoff error ', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.title('ELI', fontsize=12)
plt.show()

fig, ax = plt.subplots(1,5)
ax[0].plot(item_index, '.', label='Best arm=%s'%(best_arm)) 
ax[1].plot(oful_item_index, '.', label='Best arm=%s'%(best_arm)) 
ax[2].plot(ucb1_item_index, '.', label='Best arm=%s'%(best_arm)) 
ax[3].plot(linucb_test_item_index, '.', label='Best arm=%s'%(best_arm))
ax[4].plot(eli_item_index, '.', label='Best arm=%s'%(best_arm))  
ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# ax[2].legend(loc=4)
ax[0].set_ylabel('Item Index')
ax[0].set_title('LinUCB')
ax[1].set_title('OFUL')
ax[2].set_title('UCB1')
ax[3].set_title('LinUCB Test')
ax[4].set_title('ELI')
plt.show()

plt.figure(figsize=(5,5))
plt.plot(regret, label='LinUCB')
plt.plot(linucb_test_regret, label='LinUCB Test')
plt.plot(oful_regret, label='OFUL')
plt.plot(ucb1_regret, label='UCB1')
plt.plot(eli_regret, label='ELI')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Regret', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(beta_list, label='beta')
plt.plot(true_beta_list, label='true beta')
plt.plot(oful_beta_list, label='OFUL')
plt.plot(linucb_test_beta_list, label='LinUCB Test')
plt.plot(eli_beta_list, label='ELI')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Beta', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(error, label='LinUCB')
plt.plot(linucb_test_error, label='LinUCB Test')
plt.plot(oful_error, label='OFUL')
plt.plot(ucb1_error, label='UCB1')
plt.plot(eli_error, label='ELI')
plt.xlabel('Time', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(eli_payoff_error_matrix[:,best_arm], 'o-', markevery=0.01,color='b', label='ELI best arm')
plt.plot(payoff_error_matrix[:,best_arm], 'o-', markevery=0.01, color='k', label='LinUCB best arm')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Payoff Error', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(eli_ucb_matrix[:,best_arm], 'o-', markevery=0.01,color='b', label='ELI best arm')
plt.plot(ucb_matrix[:,worse_arm], 'o-', markevery=0.01, color='k', label='LinUCB worse arm')
plt.xlabel('Time', fontsize=12)
plt.ylabel('UCB', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(eli_beta_list, 'o-', markevery=0.01,color='b', label='ELI')
plt.plot(beta_list, 'o-', markevery=0.01, color='k', label='LinUCB')
plt.xlabel('Time', fontsize=12)
plt.ylabel('beta', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(eli_x_norm_matrix[:,best_arm], 'o-', markevery=0.01,color='b', label='ELI best arm')
plt.plot(x_norm_matrix[:,best_arm], 'o-', markevery=0.01, color='k', label='LinUCB best arm')
plt.xlabel('Time', fontsize=12)
plt.ylabel('x_norm', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].plot(ucb_matrix[:,best_arm],'o-', color='r', markevery=0.1, label='best arm') #row=0, col=0
ax[1].plot(eli_ucb_matrix[:,best_arm],'o-', color='r', markevery=0.1, label='best arm')

ax[0].plot(ucb_matrix[:, worse_arm],'o-', color='g', markevery=0.1, label='worse arm') #row=0, col=0
ax[1].plot(eli_ucb_matrix[:, worse_arm],'o-', color='g', markevery=0.1, label='worse arm')
for i in range(item_num):
	if i==best_arm:
		ax[0].plot(payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[1].plot(eli_payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[0].plot(payoff_error_matrix[:,i], label='arm= %s'%(i))
		ax[1].plot(eli_payoff_error_matrix[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
ax[0].set_ylabel('LinUCB')
ax[1].set_ylabel('ELi')
plt.title('Payoff Error')
plt.show()


fig, ax = plt.subplots(1,4)
ax[0].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0') #row=0, col=0
ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		ax[0].plot(payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[1].plot(eli_payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[2].plot(oful_payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[3].plot(ucb1_payoff_error_matrix[:, i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[0].plot(payoff_error_matrix[:,i], label='arm= %s'%(i))
		ax[1].plot(eli_payoff_error_matrix[:,i], label='arm= %s'%(i))
		ax[2].plot(oful_payoff_error_matrix[:,i], label='arm= %s'%(i))
		ax[3].plot(ucb1_payoff_error_matrix[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[3].legend(loc=4)
ax[0].set_ylabel('LinUCB')
ax[1].set_ylabel('ELi')
ax[2].set_ylabel('OFUL')
ax[3].set_ylabel('UCB1')
plt.title('Payoff Error')
plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(item_index, '.', label='Best arm=%s'%(best_arm))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Arm index', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(gaps_ucb[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Gap-UCB', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()

fig, ax = plt.subplots(1,4)
ax[0].plot(item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		ax[1].plot(index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[2].plot(mean_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[3].plot(ucb_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[1].plot(index_matrix[:,i], label='arm= %s'%(i))
		ax[2].plot(mean_matrix[:,i], label='arm= %s'%(i))
		ax[3].plot(ucb_matrix[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[3].legend(loc=4)
ax[0].set_ylabel('arm selected')
ax[1].set_ylabel('index=mean+ucb')
ax[2].set_ylabel('mean')
ax[3].set_ylabel('ucb')
plt.title('LinUCB')
plt.show()



fig, ax = plt.subplots(1,4)
ax[0].plot(eli_item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		ax[1].plot(eli_index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[2].plot(eli_mean_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[3].plot(eli_ucb_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[1].plot(eli_index_matrix[:,i], label='arm= %s'%(i))
		ax[2].plot(eli_mean_matrix[:,i], label='arm= %s'%(i))
		ax[3].plot(eli_ucb_matrix[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[3].legend(loc=4)
ax[0].set_ylabel('arm selected')
ax[1].set_ylabel('index=mean+ucb')
ax[2].set_ylabel('mean')
ax[3].set_ylabel('ucb')
plt.title('ELI')
plt.show()

fig, ax = plt.subplots(1,4)
ax[0].plot(linucb_test_item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
for i in range(item_num):
	if i==best_arm:
		ax[1].plot(linucb_test_index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[2].plot(linucb_test_mean_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
		ax[3].plot(linucb_test_ucb_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

	else:
		ax[1].plot(linucb_test_index_matrix[:,i], label='arm= %s'%(i))
		ax[2].plot(linucb_test_mean_matrix[:,i], label='arm= %s'%(i))
		ax[3].plot(linucb_test_ucb_matrix[:,i], label='arm= %s'%(i))
ax[0].legend(loc=4)
ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[3].legend(loc=4)
ax[0].set_ylabel('arm selected')
ax[1].set_ylabel('index=mean+ucb')
ax[2].set_ylabel('mean')
ax[3].set_ylabel('ucb')
plt.title('LinUCB TEST')
plt.show()


# fig, ax = plt.subplots(1,4)
# ax[0].plot(ucb1_item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# # ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# # ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# # ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# for i in range(item_num):
# 	if i==best_arm:
# 		ax[1].plot(ucb1_index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[2].plot(ucb1_mean_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[3].plot(ucb1_ucb_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

# 	else:
# 		ax[1].plot(ucb1_index_matrix[:,i], label='arm= %s'%(i))
# 		ax[2].plot(ucb1_mean_matrix[:,i], label='arm= %s'%(i))
# 		ax[3].plot(ucb1_ucb_matrix[:,i], label='arm= %s'%(i))
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# # ax[2].legend(loc=4)
# # ax[3].legend(loc=4)
# ax[0].set_ylabel('arm selected')
# ax[1].set_ylabel('index=mean+ucb')
# ax[2].set_ylabel('mean')
# ax[3].set_ylabel('ucb')
# plt.title('UCB1')
# plt.show()


# plt.figure()
# plt.plot(ucb1_est_gaps_ucb)
# plt.show()


# fig, ax = plt.subplots(1,4)
# ax[0].plot(oful_item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# for i in range(item_num):
# 	if i==best_arm:
# 		ax[1].plot(oful_gaps_ucb[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[2].plot(oful_best_index[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[3].plot(oful_est_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

# 	else:
# 		ax[1].plot(oful_gaps_ucb[:,i], label='arm= %s'%(i))
# 		ax[2].plot(oful_best_index[:,i], label='arm= %s'%(i))
# 		ax[3].plot(oful_est_gaps_ucb[:,i], label='arm= %s'%(i))
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# # ax[2].legend(loc=4)
# # ax[3].legend(loc=4)
# ax[0].set_ylabel('arm selected')
# ax[1].set_ylabel('gaps-ucb-ucb')
# ax[2].set_ylabel('best-index')
# ax[3].set_ylabel('est_gap-ucb-ucb')
# plt.title('OFUL')
# plt.show()



# fig, ax = plt.subplots(1,4)
# ax[0].plot(ucb1_item_index, '.', label='Best arm=%s'%(best_arm)) #row=0, col=0
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[3].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# for i in range(item_num):
# 	if i==best_arm:
# 		ax[1].plot(ucb1_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[2].plot(ucb1_best_index[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[3].plot(ucb1_est_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

# 	else:
# 		ax[1].plot(ucb1_gaps_ucb[:,i], label='arm= %s'%(i))
# 		ax[2].plot(ucb1_best_index[:,i], label='arm= %s'%(i))
# 		ax[3].plot(ucb1_est_gaps_ucb[:,i], label='arm= %s'%(i))
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# # ax[2].legend(loc=4)
# # ax[3].legend(loc=4)
# ax[0].set_ylabel('arm selected')
# ax[1].set_ylabel('gaps-ucb-ucb')
# ax[2].set_ylabel('best-index')
# ax[3].set_ylabel('est_gap-ucb-ucb')
# plt.title('UCB1')
# plt.show()







# fig, ax = plt.subplots(1,3)
# for i in range(item_num):
# 	if i==best_arm:
# 		ax[0].plot(index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[1].plot(oful_index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[2].plot(ucb1_index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

# 	else:
# 		ax[0].plot(index_matrix[:,i], label='arm= %s'%(i))
# 		ax[1].plot(oful_index_matrix[:,i], label='arm= %s'%(i))
# 		ax[2].plot(ucb1_index_matrix[:,i], label='arm= %s'%(i))
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[0].set_ylabel('Index')
# ax[0].set_title('LinUCB')
# ax[1].set_title('OFUL')
# ax[2].set_title('UCB1')
# plt.show()

# fig, ax = plt.subplots(1,3)
# ax[1].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[2].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# ax[0].plot(np.zeros(iteration),'o-', color='r', markevery=0.1, label='0')
# for i in range(item_num):
# 	if i==best_arm:
# 		ax[0].plot(gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[1].plot(oful_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 		ax[2].plot(ucb1_gaps_ucb[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))

# 	else:
# 		ax[0].plot(gaps_ucb[:,i], label='arm= %s'%(i))
# 		ax[1].plot(oful_gaps_ucb[:,i], label='arm= %s'%(i))
# 		ax[2].plot(ucb1_gaps_ucb[:,i], label='arm= %s'%(i))
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# ax[2].legend(loc=4)
# ax[0].set_ylabel('Gaps-2UCB')
# ax[0].set_title('LinUCB')
# ax[1].set_title('OFUL')
# ax[2].set_title('UCB1')
# plt.show()




# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(index_matrix[:,i],'*-',markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(index_matrix[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Index of arm', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()

# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(mean_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(mean_matrix[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('est mean of arm', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()

# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(x_norm_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(x_norm_matrix[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('x norm of arm', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(beta_list, label='beta')
# plt.plot(true_beta_list, label='true beta')
# plt.plot(oful_beta_list, label='OFUL')
# #plt.plot(ucb1_beta_list, label='UCB1')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Beta', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(ucb_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(ucb_matrix[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('ucb matrix', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('LinUCB')
# plt.show()


# plt.figure(figsize=(5,5))
# for i in range(item_num):
# 	if i==best_arm:
# 		plt.plot(oful_ucb_matrix[:,i],'*-', markevery=0.01, color='k', label='arm= %s'%(i))
# 	else:
# 		plt.plot(oful_ucb_matrix[:,i], label='arm= %s'%(i))
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('ucb matrix', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('OFUL')
# plt.show()


