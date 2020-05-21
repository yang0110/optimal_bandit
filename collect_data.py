import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
from utils import *
path='../results/'
# np.random.seed(2018)

phase_num=13
iteration=2**phase_num

lse_mean=np.load(path+'lse_regret_mean_item_num_10_phase_num_13.npy')
lse_std=np.load(path+'lse_regret_std_item_num_10_phase_num_13.npy')
eli_mean=np.load(path+'eli_regret_mean_item_num_10_phase_num_13.npy')
eli_std=np.load(path+'eli_regret_std_item_num_10_phase_num_13.npy')

x=range(iteration)
color_list=matplotlib.cm.get_cmap(name='tab10', lut=None).colors

plt.figure(figsize=(5,5))
plt.plot(x, eli_mean, '-o', color='r', markevery=0.1, linewidth=2, markersize=8, label='Successive Elimination')
plt.fill_between(x, eli_mean-eli_std*0.95, eli_mean+eli_std*0.95, color='r', alpha=0.2)
plt.plot(x, lse_mean, '-o', color='orange', markevery=0.1, linewidth=2, markersize=8, label='LSE')
plt.fill_between(x, lse_mean-lse_std*0.95,lse_mean+lse_std*0.95, color='orange', alpha=0.2)
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.tight_layout()
plt.savefig(path+'regret_shadow_eli_lse'+'.png', dpi=300)
plt.show()