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



user_num=1
item_num=20
dimension=5
phase_num=12
iteration=2**phase_num
sigma=0.01# noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.01
weight1=0.01
beta_online=1
gamma=0
time_width=20
loop_num=10


# beta_mean_5=np.load(path+'online_beta_mean_item_20_d_5_t_11_wind_20_beta_1.npy')
# beta_std_5=np.load(path+'online_beta_std_item_20_d_5_t_11_wind_20_beta_1.npy')

# beta_mean_10=np.load(path+'online_beta_mean_item_20_d_10_t_11_wind_20_beta_1.npy')
# beta_std_10=np.load(path+'online_beta_std_item_20_d_10_t_11_wind_20_beta_1.npy')

# beta_mean_15=np.load(path+'online_beta_mean_item_20_d_15_t_11_wind_20_beta_1.npy')
# beta_std_15=np.load(path+'online_beta_std_item_20_d_15_t_11_wind_20_beta_1.npy')


# regret_mean_5=np.load(path+'online_regret_mean_item_20_d_5_t_11_wind_20_beta_1.npy')
# regret_std_5=np.load(path+'online_regret_std_item_20_d_5_t_11_wind_20.npy')

# regret_mean_10=np.load(path+'online_regret_mean_item_20_d_10_t_11_wind_20_beta_1.npy')
# regret_std_10=np.load(path+'online_regret_std_item_20_d_10_t_11_wind_20_beta_1.npy')

# regret_mean_15=np.load(path+'online_regret_mean_item_20_d_15_t_11_wind_20_beta_1.npy')
# regret_std_15=np.load(path+'online_regret_std_item_20_d_15_t_11_wind_20_beta_1.npy')

# l_mean_5=np.load(path+'online_l_mean_item_20_d_5_t_11_wind_20_beta_1.npy')
# l_std_5=np.load(path+'online_l_std_item_20_d_5_t_11_wind_20_beta_1.npy')

# l_mean_10=np.load(path+'online_l_mean_item_20_d_10_t_11_wind_20_beta_1.npy')
# l_std_10=np.load(path+'online_l_std_item_20_d_10_t_11_wind_20_beta_1.npy')

# l_mean_15=np.load(path+'online_l_mean_item_20_d_15_t_11_wind_20_beta_1.npy')
# l_std_15=np.load(path+'online_l_std_item_20_d_15_t_11_wind_20_beta_1.npy')


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




beta_mean_10_20_1=np.load(path+'online_beta_mean_item_20_d_10_t_12_wind_20_beta_1.npy')
beta_std_10_20_1=np.load(path+'online_beta_std_item_20_d_10_t_12_wind_20_beta_1.npy')

beta_mean_10_20_2=np.load(path+'online_beta_mean_item_20_d_10_t_12_wind_20_beta_2.npy')
beta_std_10_20_2=np.load(path+'online_beta_std_item_20_d_10_t_12_wind_20_beta_2.npy')

beta_mean_10_20_3=np.load(path+'online_beta_mean_item_20_d_10_t_12_wind_20_beta_3.npy')
beta_std_10_20_3=np.load(path+'online_beta_std_item_20_d_10_t_12_wind_20_beta_3.npy')


regret_mean_10_20_1=np.load(path+'online_regret_mean_item_20_d_10_t_12_wind_20_beta_1.npy')
regret_std_10_20_1=np.load(path+'online_regret_std_item_20_d_10_t_12_wind_20_beta_1.npy')

regret_mean_10_20_2=np.load(path+'online_regret_mean_item_20_d_10_t_12_wind_20_beta_2.npy')
regret_std_10_20_2=np.load(path+'online_regret_std_item_20_d_10_t_12_wind_20_beta_2.npy')

regret_mean_10_20_3=np.load(path+'online_regret_mean_item_20_d_10_t_12_wind_20_beta_3.npy')
regret_std_10_20_3=np.load(path+'online_regret_std_item_20_d_10_t_12_wind_20_beta_3.npy')


l_mean_10_20_1=np.load(path+'online_l_mean_item_20_d_10_t_12_wind_20_beta_1.npy')
l_std_10_20_1=np.load(path+'online_l_std_item_20_d_10_t_12_wind_20_beta_1.npy')

l_mean_10_20_2=np.load(path+'online_l_mean_item_20_d_10_t_12_wind_20_beta_2.npy')
l_std_10_20_2=np.load(path+'online_l_std_item_20_d_10_t_12_wind_20_beta_2.npy')

l_mean_10_20_3=np.load(path+'online_l_mean_item_20_d_10_t_12_wind_20_beta_3.npy')
l_std_10_20_3=np.load(path+'online_l_std_item_20_d_10_t_12_wind_20_beta_3.npy')



p_mean_10_20_1=np.load(path+'online_p_mean_item_20_d_10_t_12_wind_20_beta_1.npy')
p_std_10_20_1=np.load(path+'online_p_std_item_20_d_10_t_12_wind_20_beta_1.npy')

p_mean_10_20_2=np.load(path+'online_p_mean_item_20_d_10_t_12_wind_20_beta_2.npy')
p_std_10_20_2=np.load(path+'online_p_std_item_20_d_10_t_12_wind_20_beta_2.npy')

p_mean_10_20_3=np.load(path+'online_p_mean_item_20_d_10_t_12_wind_20_beta_3.npy')
p_std_10_20_3=np.load(path+'online_p_std_item_20_d_10_t_12_wind_20_beta_3.npy')

g_mean_10_20_1=np.load(path+'online_g_mean_item_20_d_10_t_12_wind_20_beta_1.npy')
g_std_10_20_1=np.load(path+'online_g_std_item_20_d_10_t_12_wind_20_beta_1.npy')

g_mean_10_20_2=np.load(path+'online_g_mean_item_20_d_10_t_12_wind_20_beta_2.npy')
g_std_10_20_2=np.load(path+'online_g_std_item_20_d_10_t_12_wind_20_beta_2.npy')

g_mean_10_20_3=np.load(path+'online_g_mean_item_20_d_10_t_12_wind_20_beta_3.npy')
g_std_10_20_3=np.load(path+'online_g_std_item_20_d_10_t_12_wind_20_beta_3.npy')

x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, beta_mean_10_20_1,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=1, tau=20')
plt.fill_between(x, beta_mean_10_20_1-beta_std_10_20_1*0.95, beta_mean_10_20_1+beta_std_10_20_1*0.95, color='c', alpha=0.2)
plt.plot(x, beta_mean_10_20_2,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=2, tau=20')
plt.fill_between(x, beta_mean_10_20_2-beta_std_10_20_2*0.95, beta_mean_10_20_2+beta_std_10_20_2*0.95, color='y', alpha=0.2)
plt.plot(x, beta_mean_10_20_3,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=10, beta=3, tau=20')
plt.fill_between(x, beta_mean_10_20_3-beta_std_10_20_3*0.95, beta_mean_10_20_3+beta_std_10_20_3*0.95, color='r', alpha=0.2)
plt.ylim([0,5])
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Beta', fontsize=14)
plt.tight_layout()
plt.savefig(path+'real_online_beta_d_10_init_beta_1_2_3_tau_20'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(x, regret_mean_10_20_1,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=1, tau=20')
plt.fill_between(x, regret_mean_10_20_1-regret_std_10_20_1*0.95, regret_mean_10_20_1+regret_std_10_20_1*0.95, color='c', alpha=0.2)
plt.plot(x, regret_mean_10_20_2,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=2, tau=20')
plt.fill_between(x, regret_mean_10_20_2-regret_std_10_20_2*0.95, regret_mean_10_20_2+regret_std_10_20_2*0.95, color='y', alpha=0.2)
plt.plot(x, regret_mean_10_20_3,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=10, beta=3, tau=20')
plt.fill_between(x, regret_mean_10_20_3-regret_std_10_20_3*0.95, regret_mean_10_20_3+regret_std_10_20_3*0.95, color='r', alpha=0.2)
# plt.ylim([1,5])
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'real_online_regret_d_10_init_beta_1_2_3_tau_20'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, l_mean_10_20_1,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=1, tau=20')
plt.fill_between(x, l_mean_10_20_1-l_std_10_20_1*0.95, l_mean_10_20_1+l_std_10_20_1*0.95, color='c', alpha=0.2)
plt.plot(x, l_mean_10_20_2,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=2, tau=20')
plt.fill_between(x, l_mean_10_20_2-l_std_10_20_2*0.95, l_mean_10_20_2+l_std_10_20_2*0.95, color='y', alpha=0.2)
plt.plot(x, l_mean_10_20_3,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=10, beta=3, tau=20')
plt.fill_between(x, l_mean_10_20_3-l_std_10_20_3*0.95, l_mean_10_20_3+l_std_10_20_3*0.95, color='r', alpha=0.2)
# plt.ylim([1,5])
plt.legend(loc=4, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Num of suboptimal arm', fontsize=14)
plt.tight_layout()
plt.savefig(path+'real_online_l_set_d_10_init_beta_1_2_3_tau_20'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, p_mean_10_20_1,  '-.', color='c', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=1, tau=20')
plt.fill_between(x, p_mean_10_20_1-p_std_10_20_1*0.95, p_mean_10_20_1+p_std_10_20_1*0.95, color='c', alpha=0.2)
plt.plot(x, p_mean_10_20_2,  '-p',color='y', markevery=0.1, markersize=5, linewidth=2, label='d=10, beta=2, tau=20')
plt.fill_between(x, p_mean_10_20_2-p_std_10_20_2*0.95, p_mean_10_20_2+p_std_10_20_2*0.95, color='y', alpha=0.2)
plt.plot(x, p_mean_10_20_3,  '-s',color='r', markevery=0.1, markersize=5,  linewidth=2, label='d=10, beta=3, tau=20')
plt.fill_between(x, p_mean_10_20_3-p_std_10_20_3*0.95, p_mean_10_20_3+p_std_10_20_3*0.95, color='r', alpha=0.2)
# plt.ylim([1,5])
plt.legend(loc=4, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Prob of best arm', fontsize=14)
plt.tight_layout()
plt.savefig(path+'real_online_prob_d_10_init_beta_1_2_3_tau_20'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x, g_mean_10_20_3,  '-', color='r', markevery=0.2, markersize=5,  linewidth=1, label='d=10, beta=3, tau=20')
plt.fill_between(x, g_mean_10_20_3-g_std_10_20_3*0.95, g_mean_10_20_3+g_std_10_20_3*0.95, color='r', alpha=0.2)
plt.plot(x, g_mean_10_20_2,  '-',color='y', markevery=0.2, markersize=5, linewidth=1, label='d=10, beta=2, tau=20')
plt.fill_between(x, g_mean_10_20_2-g_std_10_20_2*0.95, g_mean_10_20_2+g_std_10_20_2*0.95, color='y', alpha=0.2)
plt.plot(x, g_mean_10_20_1,  '-', color='c', markevery=0.2, markersize=5, linewidth=1, label='d=10, beta=1, tau=20')
plt.fill_between(x, g_mean_10_20_1-g_std_10_20_1*0.95, g_mean_10_20_1+g_std_10_20_1*0.95, color='c', alpha=0.2)

plt.legend(loc=1, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Gradient', fontsize=14)
plt.tight_layout()
plt.savefig(path+'real_online_gradient_d_10_init_beta_1_2_3_tau_20'+'.png', dpi=100)
plt.show()







