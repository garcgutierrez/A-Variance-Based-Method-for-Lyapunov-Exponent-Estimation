# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:02:00 2025

@author: gutierrez
"""

# -*- coding: utf-8 -*-
"""
Lorenz System with Intrusive Polynomial Chaos for Lyapunov Exponent Estimation
Created on Feb 21, 2024
0.2331
Author: Gutierrez
"""

import numpy as np
from scipy.integrate import solve_ivp
import chaospy as cp
import matplotlib.pyplot as plt

# ---------------------- Configuration Parameters ---------------------- #

for iii in np.arange(7,10):

    A = 10
    B = 8/3
    C = 28.0
    P = 0.4
    Q = 0.0969
    R =  0.5
    H = 1.0
    # a, b, cc, p,q,r,h = 10, 8/3, 28.0, 0.4, 0.0969, 0.5, 1.0
    T_FINAL = 0.01
    N_EXPANSION = 2
    SIGMA0 = 0.001
    N_SAMPLES = 50
    N_COORDS = 100
    MC_SAMPLES = 200_000
    GH_ORDER = 20
    SEED = 12345
    
    MEANV = iii
    
    # ---------------------- Polynomial Chaos Setup ------------------------ #
    joint_dist = cp.Normal(mu=MEANV, sigma=SIGMA0)
    samples = joint_dist.sample(N_SAMPLES)
    poly_expansion, norms = cp.generate_expansion(N_EXPANSION, joint_dist, retall=True, normed=True)
    
    alpha, beta, gamma = cp.variable(3)
    phi_phi = cp.outer(poly_expansion, poly_expansion)
    phi_phi_phi = cp.outer(poly_expansion, phi_phi)
    
    E_beta_phi_phi = cp.E(beta * phi_phi, joint_dist)
    E_phi_phi_phi = cp.E(phi_phi_phi, joint_dist)
    E_phi_phi = cp.E(phi_phi, joint_dist)
    
    # ---------------------- System Dynamics ------------------------------- #
    def rhs_intrusive(t, c):
        N = c.shape[0] // 6
        i0, i1, i2, i3, i4, i5 = np.arange(N), np.arange(N, 2*N), np.arange(2*N, 3*N), np.arange(3*N, 4*N), np.arange(4*N, 5*N), np.arange(5*N, 6*N)
        # ci0_ci2 = np.outer(c[i0], c[i2])
        ci0_ci1 = np.outer(c[i0], c[i1])
        ci0_ci2 = np.outer(c[i0], c[i2])
        c1c3 = np.sum(ci0_ci2.reshape(-1) * E_phi_phi_phi[:, :], -1)
        dx1 = A*(c[i1] - c[i0]) + c[i3]
        dx2 = C*c[i0] - c[i1] - c1c3 + c[i5]
        dx3 = -B*c[i2] + np.sum(ci0_ci1.reshape(-1) * E_phi_phi_phi[:, :], -1)
        dx4 = H*c[i3] - c1c3 - c[i4]
        dx5 = Q*c[i1] - P*c[i4] - R*c[i0]
        dx6 = R*c[i1] - Q*c[i5]
        # dy = c[i0]+A*c[i1]+c[i3]#-c[i1] - np.sum(ci0_ci2.reshape(-1) * E_phi_phi_phi[:, :], -1) + C * c[i0]
        # dz =   np.sum(ci0_ci2.reshape(-1) * E_phi_phi_phi[:, :], -1) #-B * c[i2] + np.sum(ci0_ci1.reshape(-1) * E_phi_phi_phi[:, :], -1)
        # dz[0] += B
        # dw = c[i3]*C - Q*c[i2]
        
        
        
        # x1,x2,x3,x4,x5,x6 = c
        # dx1 = a*(x2-x1) + x4
        # dx2 = cc*x1 - x2 - x1*x3 + x6
        # dx3 = -b*x3 + x1*x2
        # dx4 = h*x4 - x1*x3 - x5
        # dx5 = q*x2 - p*x5 - r*x1
        # dx6 = r*x2 - q*x6
        
        
        
        return np.r_[dx1,dx2,dx3,dx4,dx5,dx6]
    
    # ---------------------- Renormalization ------------------------------- #
    def compute_renormalization(coeffs, sigma0):
        N = coeffs.shape[1] // 6
        c1 = coeffs[-1, :N] + coeffs[-1, N:2*N] + coeffs[-1, 2*N:3*N] + coeffs[-1, 3*N:4*N]+ coeffs[-1, 4*N:5*N]+ coeffs[-1, 5*N:6*N]
        gg1 = sigma0 / np.sum(c1[1:N])
    
        u_i = coeffs[-1, :N].copy()
        u_i[1:] = gg1 * coeffs[-1, 1:N]
    
        v_i = coeffs[-1, N:2*N].copy()
        v_i[1:] = gg1 * coeffs[-1, N+1:2*N]
    
        w_i = coeffs[-1, 2*N:3*N].copy()
        w_i[1:] = gg1 * coeffs[-1, 2*N+1:3*N]
        
        z_i = coeffs[-1, 3*N:4*N].copy()
        z_i[1:] = gg1 * coeffs[-1, 3*N+1:4*N]
        
        
        z2_i = coeffs[-1, 4*N:5*N].copy()
        z2_i[1:] = gg1 * coeffs[-1, 4*N+1:5*N]
        
        z3_i = coeffs[-1, 5*N:6*N].copy()
        z3_i[1:] = gg1 * coeffs[-1, 5*N+1:6*N]
        
        return np.r_[u_i, v_i, w_i,z_i,z2_i,z3_i]
    
    # ---------------------- Gaussian Radius Estimation -------------------- #
    def mean_radius_gaussian(Sigma, method="mc", mc_samples=MC_SAMPLES, gh_order=GH_ORDER, seed=SEED):
        Sigma = np.asarray(Sigma, float)
        d = Sigma.shape[0]
        eigvals = np.clip(np.linalg.eigvalsh(Sigma), 0, None)
    
        if method == "mc":
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal(size=(mc_samples, d))
            quad = (Z * Z) @ eigvals
            return np.mean(np.sqrt(quad))
    
        elif method == "gh":
            if d > 3:
                raise ValueError("Gauss-Hermite only implemented for d <= 3.")
            xi, wi = np.polynomial.hermite.hermgauss(gh_order)
            z1d = np.sqrt(2) * xi
            w1d = wi / np.sqrt(np.pi)
            z1d_sq = z1d ** 2
    
            acc = 0.0
            for i in range(gh_order):
                for j in range(gh_order if d > 1 else 1):
                    for k in range(gh_order if d > 2 else 1):
                        val = np.sqrt(sum(eigvals[n] * z1d_sq[[i, j, k][n]] for n in range(d)))
                        acc += np.prod([w1d[n] for n in range(d)]) * val
            return acc
    
        else:
            raise ValueError("Method must be 'mc' or 'gh'.")
    
    # ---------------------- Main Simulation Loop -------------------------- #
    def run_simulation(iterations=500000):
        initial_condition = np.r_[cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms] 
        N = initial_condition.shape[0] // 6
        coordinates = np.linspace(0, T_FINAL, N_COORDS)
        LVec = []
    
        for i in range(iterations):
            sol = solve_ivp(rhs_intrusive, [0, T_FINAL], initial_condition, t_eval=coordinates, rtol=1e-6, atol=1e-6, method='Radau')
            coeffs = sol.y.T
    
            u1 = cp.sum(poly_expansion * coeffs[:, :N], -1)
            u2 = cp.sum(poly_expansion * coeffs[:, N:2*N], -1)
            u3 = cp.sum(poly_expansion * coeffs[:, 2*N:3*N], -1)
            u4 = cp.sum(poly_expansion * coeffs[:, 3*N:4*N], -1)
            u5 = cp.sum(poly_expansion * coeffs[:, 4*N:5*N], -1)
            u6 = cp.sum(poly_expansion * coeffs[:, 5*N:6*N], -1)
            # Sigma0 = cp.Cov([u1[0], u2[0], u3[0]], joint_dist)
            # Sigma = cp.Cov([u1[-1], u2[-1], u3[-1]], joint_dist)
    
            # d0 = mean_radius_gaussian(Sigma0, method="gh")
            # d1 = mean_radius_gaussian(Sigma, method="gh")
    
            # distance = np.log(d1 / d0)
            # slope = distance / coordinates[-1]
            
    
            variance = abs(cp.Var(u1, joint_dist))
            variance2 = abs(cp.Var(u2, joint_dist))
            variance3 = abs(cp.Var(u3, joint_dist))
            variance4 = abs(cp.Var(u4, joint_dist))
            variance5 = abs(cp.Var(u5, joint_dist))
            variance6 = abs(cp.Var(u6, joint_dist))
            dkappa= np.sqrt((variance + variance2 + variance3 + variance4 + variance5 + variance6))
    
            dkappa = dkappa/dkappa[0]
            distance1 = np.log(dkappa/dkappa[0])
            
            slope = distance1[-1]/coordinates[-1]
            
            
            
            
            
            LVec.append(slope)
    
            if i % 200 == 0:
                print(f"Iteration {i}: Lyapunov = {slope:.6f}, Mean = {np.mean(LVec):.6f}")
    
            initial_condition = compute_renormalization(coeffs, SIGMA0)
    
        return LVec
    
    # ---------------------- Execution ------------------------------------- #
    
    lyapunov_values = run_simulation()
    np.savez('RosslerHyperChaos_{}_v2'.format(iii), lyapunov_values=lyapunov_values, mean=MEANV,T_FINAL=T_FINAL, SIGMA0=SIGMA0, MEANV=MEANV )
    # T_FINAL = 0.05
    # N_EXPANSION = 2
    # SIGMA0 = 0.05
    # plt.plot(lyapunov_values)
    # plt.title("Lyapunov Exponent Evolution")
    # plt.xlabel("Iteration")
    # plt.ylabel("Exponent")
    # plt.grid(True)
    # plt.show()
    
    
    