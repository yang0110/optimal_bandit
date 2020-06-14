import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
from linucb import LINUCB
from eliminator import ELI
from lse import LSE 
from lints import LINTS
from suplinucb import SupLinUCB
from spectral_eliminator import Spec_Eli
from supcbglm import SupCB_GLM
from ucbglm import UCB_GLM
from utils import *
path='../results/movielens/'
data_path='../data/movielens/'
# np.random.seed(2018)

item_num=30
dimension=18
phase_num=11
iteration=2**phase_num
alpha=1
sigma=0.1
delta=0.1
beta=1
v=0.5
loop=10

linucb_matrix=np.zeros((loop, iteration))
lints_matrix=np.zeros((loop, iteration))
eli_matrix=np.zeros((loop, iteration))
lse_matrix=np.zeros((loop, iteration))
sup_matrix=np.zeros((loop, iteration))
spec_matrix=np.zeros((loop, iteration))
glm_matrix=np.zeros((loop, iteration))
ucb_glm_matrix=np.zeros((loop, iteration))


linucb_error_matrix=np.zeros((loop, iteration))
lints_error_matrix=np.zeros((loop, iteration))
eli_error_matrix=np.zeros((loop, iteration))
lse_error_matrix=np.zeros((loop, iteration))
sup_error_matrix=np.zeros((loop, iteration))
spec_error_matrix=np.zeros((loop, iteration))
glm_error_matrix=np.zeros((loop, iteration))
ucb_glm_error_matrix=np.zeros((loop,iteration))



up_eli=np.zeros((item_num, iteration))
up_lse=np.zeros((item_num, iteration))
up_sup=np.zeros((item_num, iteration))
up_spec=np.zeros((item_num, iteration))
up_glm=np.zeros((item_num, iteration))

lo_eli=np.zeros((item_num, iteration))
lo_lse=np.zeros((item_num, iteration))
lo_sup=np.zeros((item_num, iteration))
lo_spec=np.zeros((item_num, iteration))
lo_glm=np.zeros((item_num, iteration))

item_f=pd.read_csv(data_path+'user_feature.csv', sep='\t', header=0)
user_f=pd.read_csv(data_path+'action_context.csv', sep='\t', header=0)

for l in range(loop):

	random_item=np.random.choice(range(50), size=item_num)
	random_user=np.random.choice(range(50))
	item_feature=item_f.values[random_item, 1:]
	user_feature=user_f.values[random_user, 2:]
	true_payoffs=np.dot(item_feature, user_feature)
	best_arm=np.argmax(true_payoffs)


	linucb_model=LINUCB(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	ucb_glm_model=UCB_GLM(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	lints_model=LINTS(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, v)


	eli_model=ELI(dimension, phase_num, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	spec_model=Spec_Eli(dimension, phase_num, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	sup_model=SupLinUCB(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	glm_model=SupCB_GLM(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)
	lse_model=LSE(dimension, iteration, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma, beta)


	linucb_regret, linucb_error=linucb_model.run()
	lints_regret, lints_error=lints_model.run()
	eli_regret, eli_error, eli_upper_matrix, eli_lower_matrix=eli_model.run()
	lse_regret, lse_error, lse_upper_matrix, lse_lower_matrix=lse_model.run()
	sup_regret, sup_error, sup_upper_matrix, sup_lower_matrix=sup_model.run()
	spec_regret, spec_error, spec_upper_matrix, spec_lower_matrix=spec_model.run()
	glm_regret, glm_error, glm_upper_matrix, glm_lower_matrix=glm_model.run()
	ucb_glm_regret, ucb_glm_error=ucb_glm_model.run()


	linucb_matrix[l]=linucb_regret
	lints_matrix[l]=lints_regret
	eli_matrix[l]=eli_regret
	lse_matrix[l]=lse_regret
	sup_matrix[l]=sup_regret
	spec_matrix[l]=spec_regret
	glm_matrix[l]=glm_regret
	ucb_glm_matrix[l]=ucb_glm_regret

	linucb_error_matrix[l]=linucb_error
	lints_error_matrix[l]=lints_error
	eli_error_matrix[l]=eli_error
	lse_error_matrix[l]=lse_error
	sup_error_matrix[l]=sup_error
	spec_error_matrix[l]=spec_error
	glm_error_matrix[l]=glm_error
	ucb_glm_error_matrix[l]=ucb_glm_error


	up_eli+=eli_upper_matrix
	up_lse+=lse_upper_matrix
	up_sup+=sup_upper_matrix
	up_spec+=spec_upper_matrix
	up_glm+=glm_upper_matrix


	lo_eli+=eli_lower_matrix
	lo_lse+=lse_lower_matrix
	lo_sup+=sup_lower_matrix
	lo_spec+=spec_lower_matrix
	lo_glm+=glm_lower_matrix


linucb_mean=np.mean(linucb_matrix, axis=0)
lints_mean=np.mean(lints_matrix, axis=0)
eli_mean=np.mean(eli_matrix, axis=0)
lse_mean=np.mean(lse_matrix, axis=0)
sup_mean=np.mean(sup_matrix, axis=0)
spec_mean=np.mean(spec_matrix, axis=0)
glm_mean=np.mean(glm_matrix, axis=0)
ucb_glm_mean=np.mean(ucb_glm_matrix, axis=0)

linucb_std=linucb_matrix.std(0)
lints_std=lints_matrix.std(0)
eli_std=eli_matrix.std(0)
lse_std=lse_matrix.std(0)
sup_std=sup_matrix.std(0)
spec_std=spec_matrix.std(0)
glm_std=glm_matrix.std(0)
ucb_glm_std=ucb_glm_matrix.std(0)

linucb_error_mean=np.mean(linucb_error_matrix, axis=0)
lints_error_mean=np.mean(lints_error_matrix, axis=0)
eli_error_mean=np.mean(eli_error_matrix, axis=0)
lse_error_mean=np.mean(lse_error_matrix, axis=0)
sup_error_mean=np.mean(sup_error_matrix, axis=0)
spec_error_mean=np.mean(spec_error_matrix, axis=0)
glm_error_mean=np.mean(glm_error_matrix, axis=0)
ucb_glm_error_mean=np.mean(ucb_glm_error_matrix, axis=0)


eli_upper_mean=up_eli/loop
lse_upper_mean=up_lse/loop
sup_upper_mean=up_sup/loop
spec_upper_mean=up_spec/loop
glm_upper_mean=up_glm/loop


eli_lower_mean=lo_eli/loop
lse_lower_mean=lo_lse/loop
sup_lower_mean=lo_sup/loop
spec_lower_mean=lo_spec/loop
glm_lower_mean=lo_glm/loop

plt.figure(figsize=(5,5))
plt.plot(linucb_mean,linewidth=2, label='LinUCB')
plt.plot(lints_mean, linewidth=2,label='LinTS')
plt.plot(ucb_glm_mean, linewidth=2, label='UCB-GLM')
plt.plot(lse_mean, linewidth=2,label='LSE')
plt.plot(eli_mean, linewidth=2,label='Phased')
plt.plot(sup_mean, linewidth=2,label='SupLinUCB')
plt.plot(spec_mean, linewidth=2, label='Spectral')
plt.plot(glm_mean, linewidth=2, label='SupCB-GLM')
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'cumulative_regret'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error_mean,'-.', markevery=0.1, linewidth=2, label='LinUCB')
plt.plot(lints_error_mean+0.2, '-.', markevery=0.1,linewidth=2, label='LinTS')
plt.plot(ucb_glm_error_mean+0.4, '-.', markevery=0.1,linewidth=2, label='UCB-GLM')
plt.plot(lse_error_mean+0.6, '-.', markevery=0.1,linewidth=2, label='LSE')
plt.plot(eli_error_mean+0.8, '-.', markevery=0.1,linewidth=2, label='Phased')
plt.plot(sup_error_mean+1.0, '-.', markevery=0.1, linewidth=2,label='SupLinUCB')
plt.plot(spec_error_mean+1.2, '-.', markevery=0.1,linewidth=2, label='Spectral')
plt.plot(glm_error_mean+1.4, '-.', markevery=0.1,linewidth=2, label='SupCB-GLM')
plt.legend(loc=1, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Reward Error', fontsize=14)
plt.tight_layout()
plt.savefig(path+'reward_error'+'.png', dpi=100)
plt.show()


color_list=matplotlib.cm.get_cmap(name='tab20', lut=None).colors

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(sup_upper_mean[i], color=color_list[i],  linewidth=2)
	plt.plot(sup_lower_mean[i], '-.', color=color_list[i], linewidth=2, markevery=0.2)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence Interval', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('SupLinUCB', fontsize=14)
plt.tight_layout()
plt.savefig(path+'suplinucb_confidence_interval'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(glm_upper_mean[i], color=color_list[i],  linewidth=2)
	plt.plot(glm_lower_mean[i], '-.', color=color_list[i], linewidth=2, markevery=0.2)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence Interval', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('SupCB-GLM', fontsize=14)
plt.tight_layout()
plt.savefig(path+'supcbglm_confidence_interval'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(eli_upper_mean[i], color=color_list[i],  linewidth=2)
	plt.plot(eli_lower_mean[i], '-.', color=color_list[i], linewidth=2, markevery=0.2)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence Interval', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('Phased Eliminator', fontsize=14)
plt.tight_layout()
plt.savefig(path+'eliminator_confidence_interval'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(spec_upper_mean[i], color=color_list[i],  linewidth=2)
	plt.plot(spec_lower_mean[i], '-.', color=color_list[i], linewidth=2, markevery=0.2)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence Interval', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('Spectral Eliminator', fontsize=14)
plt.tight_layout()
plt.savefig(path+'spectral_eliminator_confidence_interval'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
for i in range(item_num):
	plt.plot(lse_upper_mean[i], color=color_list[i],  linewidth=2)
	plt.plot(lse_lower_mean[i], '-.', color=color_list[i], linewidth=2, markevery=0.2)
plt.ylim([-2,2])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Confidence Interval', fontsize=12)
# plt.legend(loc=1, fontsize=12)
# plt.title('LSE', fontsize=14)
plt.tight_layout()
plt.savefig(path+'lse_confidence_interval'+'.png', dpi=100)
plt.show()














