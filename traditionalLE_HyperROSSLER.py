# -*- coding: utf-8 -*-
"""
Largest Lyapunov Exponent (LLE) for Lorenz-63 with:
  a (sigma) = 16
  b (beta)  = 4
  cc (rho)  = 45.92

Two standard methods:
  1) Benettin (variational) using analytical Jacobian
  2) Wolf (two-trajectory) Jacobian-free

Dependencies:
  - numpy
  - scipy (solve_ivp)
  - matplotlib (optional, for plotting)


0.2331


Author: (you)
"""

import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

# Optional plotting (set PLOT=True to see moving-average of segment lambdas)
PLOT = True
if PLOT:
    import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Define Lorenz-63 system with your parameters
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z 
# ------------------------------------------------------------
a, b, cc, p,q,r,h = 10, 8/3, 28.0, 0.4, 0.0969, 0.5, 1.0

def f(t, c):
    x1,x2,x3,x4,x5,x6 = c
    dx1 = a*(x2-x1) + x4
    dx2 = cc*x1 - x2 - x1*x3 + x6
    dx3 = -b*x3 + x1*x2
    dx4 = h*x4 - x1*x3 - x5
    dx5 = q*x2 - p*x5 - r*x1
    dx6 = r*x2 - q*x6
    
    return [dx1, dx2, dx3, dx4, dx5, dx6]






# ------------------------------------------------------------
# Wolf two-trajectory method (Jacobian-free)
# ------------------------------------------------------------
def largest_lyapunov_wolf(
    f,
    x0,
    t_span=(0.0, 425.0),
    renorm_dt=0.05,
    transient=25.0,
    eps0=1e-6,
    method="DOP853",
    rtol=1e-9,
    atol=1e-12,
    progress=True,
    seed=12345,
):
    """
    Compute LLE by evolving two nearby trajectories x(t) and y(t).
    Every renorm_dt, measure d=||y-x||, accumulate log(d/eps0), then reset
    y = x + eps0 * (y-x)/||y-x||.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    t0, tf = t_span
    if transient < 0 or transient > (tf - t0):
        raise ValueError("transient must be in [0, tf - t0].")

    rng = np.random.default_rng(seed)
    d0 = rng.normal(size=n)
    d0 = d0 / (norm(d0) + 1e-300) * eps0

    # Transient
    t = t0
    x = x0.copy()
    y = x0 + d0


    # Accumulation
    logs = []
    segments = 0
    T_accum = 0.0
    while t < tf - 1e-15:
        t_next = min(t + renorm_dt, tf)
        sol_x = solve_ivp(f, (t, t_next), x, method=method, rtol=rtol, atol=atol, t_eval=[t_next])
        sol_y = solve_ivp(f, (t, t_next), y, method=method, rtol=rtol, atol=atol, t_eval=[t_next])
        x = sol_x.y[:, -1]
        y = sol_y.y[:, -1]

        d = y - x
        dnorm = norm(d)
        logs.append(np.log((dnorm + 1e-300) / eps0))
        segments += 1
        T_accum += (t_next - t)

        # Renormalize
        u = d / (dnorm + 1e-300)
        y = x + eps0 * u

        t = t_next

        if progress and (segments % 100 == 0):
            print(f"[Wolf] seg={segments}  LLE≈{np.sum(logs)/T_accum:.6f}")

    lle = np.sum(logs) / max(T_accum, 1e-300)
    return lle, {
        "segments": segments,
        "lambda_segments": np.array(logs) / renorm_dt,
        "time_used": T_accum,
    }


# ------------------------------------------------------------
# Run both methods for your parameters
# ------------------------------------------------------------
if __name__ == "__main__":
    # Initial condition (any nontrivial point works)
   

    # Integration configuration
    t_span=(0.0, 5000.0)

    # t_span    = (0.0, 100.0)  # total time
    transient = 0.0          # discard first 10 units
    renorm_dt = 0.01          # segment length

    # # --- Benettin (with analytical Jacobian) ---
    # lle_ben, info_ben = largest_lyapunov_benettin(
    #     f=f,
    #     jac=jac,                 # analytical Jacobian
    #     x0=x0,
    #     t_span=t_span,
    #     renorm_dt=renorm_dt,
    #     transient=transient,
    #     v0_norm=1e-6,
    #     method="DOP853",         # try "LSODA" if you suspect stiffness
    #     rtol=1e-9,
    #     atol=1e-12,
    #     progress=True,
    #     seed=12345,
    # )
    # print(f"\n[Benettin] LLE ≈ {lle_ben:.6f} (1/time units)")

    # --- Wolf (two-trajectory, Jacobian-free) ---
    for iii in np.arange(1,10):
        x0 = iii*np.random.rand(6)
        lle_wolf, info_wolf = largest_lyapunov_wolf(
            f=f,
            x0=x0,
            t_span=t_span,
            renorm_dt=renorm_dt,
            transient=transient,
            eps0=1e-6,
            method="Radau",
            rtol=1e-9,
            atol=1e-12,
            progress=True,
            seed=12345,
        )
        print(f"[Wolf]     LLE ≈ {lle_wolf:.6f} (1/time units)")
        np.savez('MCChaos_HYPERROSSB_{}'.format(iii), lle_wolf=lle_wolf, info_wolf=info_wolf)

    # # Optional: plot instantaneous λ per segment and moving average (Benettin)
    # if PLOT:
    #     lam_seg = info_wolf["lambda_segments"]  # per-segment instantaneous λ
    #     # plt.figure(figsize=(9, 4.8))
    #     # plt.plot(lam_seg, lw=0.8, alpha=0.85, label="λ per segment (Benettin)")
    #     # if len(lam_seg) > 10:
    #     #     window = min(200, max(5, len(lam_seg)//10))
    #     #     kernel = np.ones(window) / window
    #     #     ma = np.convolve(lam_seg, kernel, mode="same")
    #     #     plt.plot(ma, 'r', lw=2, alpha=0.9, label=f"Moving avg (window={window})")
    #     # plt.axhline(lle_wolf, color='k', ls='--', alpha=0.6,
    #     #             label=f"Mean: {lle_wolf:.6f}")
    #     # plt.xlabel("Segment index")
    #     # plt.ylabel("Instantaneous Lyapunov")
    #     # plt.title(f"Largest Lyapunov Exponent (σ={sigma}, ρ={rho}, β={beta})")
    #     # plt.legend()
    #     # plt.tight_layout()
    #     # plt.show()