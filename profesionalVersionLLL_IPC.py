# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:02:00 2025

@author: gutierrez
"""

# -*- coding: utf-8 -*-
"""
Lorenz System with Intrusive Polynomial Chaos for Lyapunov Exponent Estimation
Created on Feb 21, 2024
Author: Gutierrez
"""

import numpy as np
from scipy.integrate import solve_ivp
import chaospy as cp
import matplotlib.pyplot as plt

# ---------------------- Configuration Parameters ---------------------- #
A = 16
B = 4
C = 45.92
T_FINAL = 0.05
N_EXPANSION = 2
SIGMA0 = 0.01
N_SAMPLES = 50
N_COORDS = 100
MC_SAMPLES = 200_000
GH_ORDER = 20
SEED = 12345

# ---------------------- Configuration Parameters ---------------------- #

for iii in np.arange(2,5):

# ---------------------- Polynomial Chaos Setup ------------------------ #
    MEANV = iii*1.0
    
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
        N = c.shape[0] // 3
        i0, i1, i2 = np.arange(N), np.arange(N, 2*N), np.arange(2*N, 3*N)
        ci0_ci2 = np.outer(c[i0], c[i2])
        ci0_ci1 = np.outer(c[i0], c[i1])
    
        dx = A * (c[i1] - c[i0])
        dy = -c[i1] - np.sum(ci0_ci2.reshape(-1) * E_phi_phi_phi[:, :], -1) + C * c[i0]
        dz = -B * c[i2] + np.sum(ci0_ci1.reshape(-1) * E_phi_phi_phi[:, :], -1)
    
        return np.r_[dx, dy, dz]
    
    # ---------------------- Renormalization ------------------------------- #
    def compute_renormalization(coeffs, sigma0):
        N = coeffs.shape[1] // 3
        c1 = coeffs[-1, :N] + coeffs[-1, N:2*N] + coeffs[-1, 2*N:3*N]
        gg1 = sigma0 / np.sum(c1[1:N])
    
        u_i = coeffs[-1, :N].copy()
        u_i[1:] = gg1 * coeffs[-1, 1:N]
    
        v_i = coeffs[-1, N:2*N].copy()
        v_i[1:] = gg1 * coeffs[-1, N+1:2*N]
    
        w_i = coeffs[-1, 2*N:3*N].copy()
        w_i[1:] = gg1 * coeffs[-1, 2*N+1:3*N]
    
        return np.r_[u_i, v_i, w_i]
    
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
    def run_simulation(iterations=100000):
        initial_condition = np.r_[cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms,cp.E(alpha * poly_expansion, joint_dist) / norms] 
        N = initial_condition.shape[0] // 3
        coordinates = np.linspace(0, T_FINAL, N_COORDS)
        LVec = []
    
        for i in range(iterations):
            sol = solve_ivp(rhs_intrusive, [0, T_FINAL], initial_condition, t_eval=coordinates, rtol=1e-6, atol=1e-6, method='DOP853')
            coeffs = sol.y.T
    
            u1 = cp.sum(poly_expansion * coeffs[:, :N], -1)
            u2 = cp.sum(poly_expansion * coeffs[:, N:2*N], -1)
            u3 = cp.sum(poly_expansion * coeffs[:, 2*N:3*N], -1)
    
            # Sigma0 = cp.Cov([u1[0], u2[0], u3[0]], joint_dist)
            # Sigma = cp.Cov([u1[-1], u2[-1], u3[-1]], joint_dist)
    
            # d0 = mean_radius_gaussian(Sigma0, method="gh")
            # d1 = mean_radius_gaussian(Sigma, method="gh")
    
            # distance = np.log(d1 / d0)
            # slope = distance / coordinates[-1]
            
    
            variance = abs(cp.Var(u1, joint_dist))
            variance2 = abs(cp.Var(u2, joint_dist))
            variance3 = abs(cp.Var(u3, joint_dist))
    
            dkappa= np.sqrt( (variance+variance2+variance3))
    
            dkappa = dkappa/dkappa[0]
            distance1 = np.log(dkappa/dkappa[0])
            
            slope = distance1[-1]/coordinates[-1]

            
            LVec.append(slope)
    
            if i % 200 == 0:
                print(f"Iteration {i}: Lyapunov = {slope:.6f}, Mean = {np.mean(LVec):.6f}")
    
            initial_condition = compute_renormalization(coeffs, SIGMA0)
    
        return LVec

    lyapunov_values = run_simulation()
    np.savez('LorenzChaos_{}_N100000'.format(iii), lyapunov_values=lyapunov_values, mean=MEANV,T_FINAL=T_FINAL, SIGMA0=SIGMA0, MEANV=MEANV )
    # T_FINAL = 0.05
