# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:30:47 2025

@author: gutierrez
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

plt.rcParams.update({'font.size': 14})  # Change 14 to your desired size

# List of file names
files = [
    'LorenzChaos_1_N100000.npz',
    'LorenzChaos_2_N100000.npz',
    # 'LorenzChaos_3_N100000.npz',
    'LorenzChaos_4_N100000.npz',
    # 'LorenzChaos_0_N100000.npz'
]


files2 = ['MCChaos_1.npz',
         'MCChaos_2.npz',
         # 'MCChaos_3.npz',
         'MCChaos_4.npz',
         # 'MCChaos_0.npz'
     ]

Nfilter = 10000
# Compute mean Lyapunov exponent values with filtering
means = []
for file in files:
    try:
        data = np.load(file)
        lv = data['lyapunov_values']
        lv_filtered = uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files[0]:
            means = lv_filtered[Nfilter:]
        else:
            means = np.c_[means, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")


for file in files2:
    try:
        data = np.load(file,allow_pickle=True)
        lv = data['info_wolf'].item()['lambda_segments']
        lv_filtered = uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files2[0]:
            means2 = lv_filtered[Nfilter:]
        else:
            means2 = np.c_[means2, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")
# Compute cumulative mean
promedios = np.cumsum(means, axis=0) / np.arange(np.shape(means)[0]).reshape(-1, 1)
promedios2 = np.cumsum(means2, axis=0) / np.arange(np.shape(means2)[0]).reshape(-1, 1)
# Plotting
plt.figure(figsize=(8, 5))
colors = ['b', 'g', 'm', 'c', 'y']  # Different colors for each experiment
plt.axhline(y=1.5, color='r', linestyle='--',linewidth=4, label='True Value (1.5)')
t = np.arange(len(promedios))*0.05
for i in range(promedios.shape[1]):
    plt.plot(t, promedios[:, i], color=colors[i],linewidth=2, label=f'IPC {i+1}')
    print(np.mean(promedios2[-1]))
    plt.plot(t, promedios2[:, i], color=colors[i],linewidth=3,linestyle=':', label=f' Legacy {i+1}')

plt.ylim([1.4, 1.6])
plt.xlim([0, 4500])
plt.xlabel(r'$t$ (TU)')
plt.ylabel('Mean Lyapunov Exponent (1/TU)')
plt.title('Lorenz system')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('IPC_Lorenz.pdf')
plt.show()






plt.figure()

files = [
    'RosslerChaos_0_v2.npz',
    'RosslerChaos_1_v2.npz',
    'RosslerChaos_2_v2.npz',
    # 'RosslerChaos_3_v2.npz',
    # 'RosslerChaos_4_v2.npz',
    # 'RosslerChaos_5_v2.npz'
]


files2 = ['MCChaos_ROSSB_1.npz',
         'MCChaos_ROSSB_2.npz',
         'MCChaos_ROSSB_3.npz',
         'MCChaos_ROSSB_4.npz',
         'MCChaos_ROSSB_0.npz'
     ]

Nfilter = 10000
# Compute mean Lyapunov exponent values with filtering
means = []
for file in files:
    try:
        data = np.load(file)
        lv = data['lyapunov_values']
        lv_filtered = uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files[0]:
            means = lv_filtered[Nfilter:]
        else:
            means = np.c_[means, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")
        
for file in files2:
    try:
        data = np.load(file,allow_pickle=True)
        lv = data['info_wolf'].item()['lambda_segments']
        lv_filtered = uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files2[0]:
            means2 = lv_filtered[Nfilter:]
        else:
            means2 = np.c_[means2, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")
# Compute cumulative mean
promedios = np.cumsum(means, axis=0) / np.arange(np.shape(means)[0]).reshape(-1, 1)
promedios2 = np.cumsum(means2, axis=0) / np.arange(np.shape(means2)[0]).reshape(-1, 1)

# Plotting
plt.figure(figsize=(8, 5))
colors = ['b', 'g', 'm', 'c', 'y']  # Different colors for each experiment
t = np.arange(len(promedios))*0.05
plt.axhline(y=0.0901, color='r', linestyle='--',linewidth=4, label='True Value (0.0901)')
for i in range(promedios.shape[1]):
    plt.plot(t, promedios[:, i], color=colors[i],linewidth=2)
    print(np.mean(promedios2[-1]))
    plt.plot(t, promedios2[:, i], color=colors[i],linestyle=':',linewidth=3)


plt.ylim([0.07, 0.12])
plt.xlim([0, 4500])
plt.xlabel(r'$t$ (TU)')
plt.ylabel('Mean Lyapunov Exponent (1/TU)')
plt.title('Rossler-chaos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('IPC_Rossler.pdf')
plt.show()






plt.figure()

files = [
    'RosslerHyperChaos_6_v2.npz',
    'RosslerHyperChaos_5_v2.npz',
    'RosslerHyperChaos_4_v2.npz',
    # 'RosslerChaos_3_v2.npz',
    # 'RosslerChaos_4_v2.npz',
    # 'RosslerChaos_5_v2.npz'
]


files2 = ['MCChaos_HYPERROSSB_7.npz',
         'MCChaos_HYPERROSSB_8.npz',
         'MCChaos_HYPERROSSB_9.npz'
     ]

Nfilter = 50000
# Compute mean Lyapunov exponent values with filtering
means = []
for file in files:
    try:
        data = np.load(file)
        lv = data['lyapunov_values']
        lv_filtered = lv#uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files[0]:
            means = lv_filtered[Nfilter:]
        else:
            means = np.c_[means, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")
        
for file in files2:
    try:
        data = np.load(file,allow_pickle=True)
        lv = data['info_wolf'].item()['lambda_segments']
        lv_filtered = lv#uniform_filter1d(lv, size=100)  # Apply smoothing filter
        if file == files2[0]:
            means2 = lv_filtered[Nfilter:]
        else:
            means2 = np.c_[means2, lv_filtered[Nfilter:]]
    except FileNotFoundError:
        print(f"File not found: {file}")
# Compute cumulative mean
promedios = np.cumsum(means, axis=0) / np.arange(np.shape(means)[0]).reshape(-1, 1)
promedios2 = np.cumsum(means2, axis=0) / np.arange(np.shape(means2)[0]).reshape(-1, 1)
Njump =300
# Plotting
plt.figure(figsize=(8, 5))
colors = ['b', 'g', 'm', 'c', 'y']  # Different colors for each experiment
t = np.arange(len(promedios2))*0.01
plt.axhline(y=0.37, color='r', linestyle='--',linewidth=4, label='True Value (0.37)')
for i in range(promedios.shape[1]):
    plt.plot(t[::Njump], promedios[::Njump, i], color=colors[i],linewidth=2, label=f'IPC {i+1}')
    print(np.mean(promedios2[-1]))
    plt.plot(t[::Njump], promedios2[::Njump, i], color=colors[i],linestyle=':',linewidth=3, label=f' Legacy {i+1}')


plt.ylim([0.3, 0.44])
# plt.xlim([0, 4500])
plt.xlabel(r'$t$ (TU)')
plt.ylabel('Mean Lyapunov Exponent (1/TU)')
plt.title('Al-Azzawi et al Attractor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('IPC_HyperRosslerv3.pdf')
plt.show()