# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:34:00 2024

@author: gutierrez
"""

from pylab import *
from scipy.integrate import solve_ivp
import chaospy as cp
from matplotlib import pyplot
import numpy
import sys 
from scipy.integrate import odeint
from computeMetrics import *
import csv
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})  # Change 14 to your desired size

def right_hand_side(t,c):
    N = c.shape[0] // 3
    i0, i1, i2 = np.arange(N), np.arange(N, 2*N), np.arange(2*N, 3*N)
    ci0_ci2 = np.outer(c[i0], c[i2])
    ci0_ci1 = np.outer(c[i0], c[i1])

    dx = a * (c[i1] - c[i0])
    dy = -c[i1] - np.sum(ci0_ci2.reshape(-1) * e_phi_phi_phi[:, :], -1) + cc * c[i0]
    dz = -b * c[i2] + np.sum(ci0_ci1.reshape(-1) * e_phi_phi_phi[:, :], -1)

    return np.r_[dx, dy, dz]


def right_hand_side_det(t,c):
    lul = [a*(c[1]-c[0]),
    c[0]*(-c[2])-c[1]+cc*c[0],
     -b*c[2]+c[0]*c[1],
    ]
    return lul




a = 16
b = 4
cc = 45.92
Tfinal =0.15
NMontecarlo = 100000

for Tfinal in [0.15]:
    for NNN in [4]:
    
        Nexpansion = NNN
        
        order = NNN
        
        gauss_evals = []
        NN = 500
        coordinates = numpy.linspace(0, Tfinal, NN)
        dt = coordinates[1]-coordinates[0]
        samples = []
        
        
        joint = cp.J(cp.Normal(mu=1.0, sigma=0.1), cp.Normal(mu=2.0, sigma=0.2),cp.Normal(mu=-1.0, sigma=0.1))
        # joint = cp.J(cp.Laplace(sigma=0.1),cp.Laplace(sigma=0.1),cp.Laplace(sigma=0.1))
        Montecarlo_variables =  joint.sample(NMontecarlo)
        
        polynomial_expansion, norms = cp.generate_expansion(Nexpansion, joint, retall=True, normed=True)
        
        alpha, beta, gamma = cp.variable(3)
        
        
        phi_phi = cp.outer(
            polynomial_expansion, polynomial_expansion)
        phi_phi_phi = cp.outer(
            polynomial_expansion, phi_phi)
        
        e_beta_phi_phi = cp.E(beta*phi_phi, joint)
        e_phi_phi_phi = cp.E(phi_phi_phi, joint)
        e_phi_phi = cp.E(phi_phi, joint)
        
        
        e_g2_phi = cp.E((b*cc)*polynomial_expansion, joint)/norms
        e_alpha_phi = cp.E((alpha)*polynomial_expansion, joint)
        e_beta_phi = cp.E((beta)*polynomial_expansion, joint)
        e_gamma_phi = cp.E((gamma)*polynomial_expansion, joint)
        initial_condition = r_[e_alpha_phi/norms,e_beta_phi/norms,e_gamma_phi/norms]
        
        
        coefficients = solve_ivp(right_hand_side,[0,Tfinal],
                              initial_condition,t_eval=coordinates, rtol=1e-12, atol=1e-12)['y'].T
        
        
        N = int(shape(initial_condition)[0]/3)
        u_approx = cp.sum(polynomial_expansion*coefficients[:,:N], -1)
        u_approx2 = cp.sum(polynomial_expansion*coefficients[:,N:2*N], -1)
        u_approx3 = cp.sum(polynomial_expansion*coefficients[:,2*N:3*N], -1)
        
        
        mean1 = cp.E(u_approx, joint)
        variance1 = cp.Var(u_approx, joint)
        
        mean2 = cp.E(u_approx2, joint)
        variance2 = cp.Var(u_approx2, joint)
        
        
        mean3 = cp.E(u_approx3, joint)
        variance3 = cp.Var(u_approx3, joint)
        
        sigma = numpy.sqrt(variance1)
        sigma2 = numpy.sqrt(variance2)
        sigma3 = numpy.sqrt(variance3)
        
        
        
        perc3 = cp.Perc(u_approx3, [2.5, 97.5], joint, sample=NMontecarlo)
        perc2 = cp.Perc(u_approx2, [2.5, 97.5], joint, sample=NMontecarlo)
        perc1 = cp.Perc(u_approx, [2.5, 97.5], joint, sample=NMontecarlo)
        
        pyplot.fill_between(coordinates, perc1[0,:], perc1[1,:], alpha=0.5)
        pyplot.plot(coordinates, mean1, label=r'$x$',linewidth='2')
        
        pyplot.fill_between(coordinates, perc2[0,:], perc2[1,:], alpha=0.5)
        pyplot.plot(coordinates, mean2, label=r'$y$',linewidth='2')
        
        pyplot.fill_between(coordinates, perc3[0,:], perc3[1,:], alpha=0.5)
        pyplot.plot(coordinates, mean3, label=r'$z$',linewidth='2')
        # pyplot.show()
        
        
        # samples.append(Montecarlo_variables[0,0])
        for ii in arange(len(Montecarlo_variables[0,:])):
            print(ii)
            sample = solve_ivp(right_hand_side_det,[0,Tfinal],
                                  [Montecarlo_variables[0,ii],Montecarlo_variables[1,ii],Montecarlo_variables[2,ii]],method='Radau',t_eval=coordinates)['y'].T
            
            if(ii<20):
                pyplot.plot(coordinates, sample[:,0], '--b',linewidth='0.4')
                pyplot.plot(coordinates, sample[:,1], '--r',linewidth='0.4')
                pyplot.plot(coordinates, sample[:,2], '--g',linewidth='0.4')
            samples.append(sample)
        
        gmean = double(samples).mean(axis=0)
        gstd = np.float32(samples).std(axis=0)
        # samplesNIPC_x, samplesNIPC_y, samplesNIPC_z, mean_NI, var_NI = computeNIPC(joint, order)
        
        
        grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        tick_params(axis='both', which='major', labelsize=12)
        xlabel(r'$t$ (TU)', fontsize=15)
        ylabel('Coordinate', fontsize=12)
        legend( fontsize=12, ncol=3,loc='upper left', frameon=False)
        tight_layout()
        # print('Montecarlo: {}      PC: {} {}  {}  NI: {} {}  {}  '.format(gmean[-1], mean[-1],mean2[-1],mean3[-1],mean_NI[0], mean_NI[1], mean_NI[2]))
        # print('Montecarlo: {}     PC: {} {}  {}  NI: {} {}  {} '.format(gstd[-1], sigma[-1],sigma2[-1],sigma3[-1],sqrt(var_NI[0]), sqrt(var_NI[1]), sqrt(var_NI[2])))
        # # plt.savefig("plFigure.png", dpi=300, bbox_inches='tight')
        
        plt.savefig('TemporalEvolution_System.pdf')
        
        
        samples = double(samples)
        samplesM_x = samples[:,:,0]
        samplesM_y = samples[:,:,1]
        samplesM_z = samples[:,:,2]
        meanMC_0 = gmean[:,0]
        meanMC_1 = gmean[:,1]
        meanMC_2 = gmean[:,2]
        
        
        stdMC_0 = gstd[:,0]
        stdMC_1 = gstd[:,1]
        stdMC_2 = gstd[:,2]
        
        eR1 = abs((mean1-meanMC_0)/meanMC_0[0])
        eR2 = abs((mean2-meanMC_1)/meanMC_1[0])
        eR3 = abs((mean3-meanMC_2)/meanMC_2[0])
        # figure()
        # plot(coordinates, eR1)
        # plot(coordinates, eR2)
        # plot(coordinates, eR3)
        
        
        estdMC_0 = abs((sigma-stdMC_0)/stdMC_0[0])
        estdMC_1 = abs((sigma2-stdMC_1)/stdMC_1[0])
        estdMC_2 = abs((sigma3-stdMC_2)/stdMC_2[0])
        # plot(coordinates, stdMC_0,'--')
        # plot(coordinates, stdMC_1,'--')
        # plot(coordinates, stdMC_2,'--')
        

        # Assumes you already have:
        # coordinates, eR1, eR2, eR3   # mean-relative-error for x, y, z
        # stdMC_0, stdMC_1, stdMC_2    # std-relative-error for x, y, z
        colors = {"x":"tab:blue", "y":"tab:orange", "z":"tab:green"}
        
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': 14}) 
        ax2 = ax.twinx()  # right y-axis for std errors
        # --- Left axis: mean errors (solid) ---
        l1, = ax.plot(coordinates, eR1, color=colors["x"], lw=2, label="x - mean")
        l2, = ax.plot(coordinates, eR2, color=colors["y"], lw=2, label="y - mean")
        l3, = ax.plot(coordinates, eR3, color=colors["z"], lw=2, label="z - mean")
        
        # --- Right axis: std errors (dashed) ---
        r1, = ax2.plot(coordinates, estdMC_0, ls='--', color=colors["x"], lw=2, label="x - std")
        r2, = ax2.plot(coordinates, estdMC_1, ls='--', color=colors["y"], lw=2, label="y - std")
        r3, = ax2.plot(coordinates, estdMC_2, ls='--', color=colors["z"], lw=2, label="z - std")
        
        # Labels and titles
        ax.set_xlabel(r"$t$ (TU)")
        ax.set_ylabel("Relative error (mean)")
        ax2.set_ylabel("Relative error (std)")
        
        ax.set_title("Relative Error of Mean (left) and Std (right) for x, y, z")
        
        # Grid and layout
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # Combine legends from both axes
        lines = [l1, l2, l3, r1, r2, r3]
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, ncol=2, frameon=False, loc="upper left")
        ax.set_xlim([0,0.15])
        # If std errors are much larger, keep mean linear and make std log-scale:
        # ax2.set_yscale('log')
        
        # Optionally, give the right axis a faint spine color to distinguish
        # ax2.spines['right'].set_color('0.5')
        # ax2.tick_params(axis='y', colors='0.4')
        
        plt.tight_layout()
        plt.savefig('TemporalEvolutionError.pdf')
        plt.show()

        # bc_PC_x = computeMetrics(samplesM_x, samplesPC_x)
            # # bc_NIPC_x =computeMetrics(samplesM_x, samplesNIPC_x)
            # bc_PC_y =computeMetrics(samplesM_y, samplesPC_y)
            # # bc_NIPC_y =computeMetrics(samplesM_y, samplesNIPC_y)
            # bc_PC_z =computeMetrics(samplesM_z, samplesPC_z)
            # # bc_NIPC_z =computeMetrics(samplesM_z, samplesNIPC_z)
            
            
            # data_to_write = r_[Tfinal, Nexpansion,NMontecarlo,order,gmean[-1], mean[-1],mean2[-1],mean3[-1],mean_NI[0], mean_NI[1], mean_NI[2],gstd[-1], sigma[-1],sigma2[-1],sigma3[-1],sqrt(var_NI[0]), sqrt(var_NI[1]), sqrt(var_NI[2]),bc_PC_x,bc_NIPC_x,bc_PC_y,bc_NIPC_y,bc_PC_z,bc_NIPC_z]
            
            # filename = 'Pruebasv3conDWeibull.csv'
            
            # with open(filename, 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(data_to_write)
            #     # writer.writerow(data1)