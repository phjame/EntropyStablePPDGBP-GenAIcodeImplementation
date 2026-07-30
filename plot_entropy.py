"""
plot_entropy.py
================

Runs the coupled 1Dx-2Dp Boltzmann-Poisson diode simulation with the final,
validated bp_dg.py solver, and tracks the (semi-discrete) entropy norm of
Theorem 2.1:

    E(t) = int f_h(x,p,mu,t)^2 * exp(H(x,p,t)) * p^2  dp dmu dx,
    H(x,p,t) = eps(p) - q*Phi(x,t).

Theorem 2.1 / Corollary 2.2 / Theorem 2.4 claim this quantity is
non-increasing in time. This script plots both the raw entropy norm E(t)
and the *relative* entropy E(t)/E(0), which is the more diagnostic quantity
for checking the stability claim (should stay <= 1 and be monotonically
non-increasing).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bp_dg import (
    Mesh, Physics, project_function, ssp_rk3_step, estimate_dt,
    charge_density_cellavg, solve_poisson_periodic,
    solve_poisson_potential_periodic, entropy_norm,
)


def main():
    # Same setup as run_diode_demo.py, kept modest since the collision
    # quadrature (NQC) and the entropy-norm diagnostic itself are the most
    # expensive parts of this reference implementation.
    mesh = Mesh(L=1.0, p_min=0.05, p_max=2.5, Nx=8, Np=8, Nmu=4)
    phys = Physics(q=1.0, eps_permittivity=1.0, m_star=1.0,
                    hbar_omega=0.3, K_scatter=1.0, T_lattice=1.0)

    p0, sigma = 0.9, 0.35

    def f0(x, p, mu):
        return np.exp(-((p - p0) ** 2) / (2 * sigma ** 2))

    field = project_function(mesh, f0)
    rho0 = charge_density_cellavg(mesh, field)
    field *= 0.4 / np.sum(rho0 * mesh.hx)

    N_cells = 0.2 + 0.3 * (mesh.xc < 0.2) + 0.3 * (mesh.xc > 0.8)

    NSTEPS = 40
    t_hist = [0.0]
    entropy_hist = []
    mass_hist = []

    f = field.copy()
    t = 0.0
    for step in range(NSTEPS):
        rho = charge_density_cellavg(mesh, f)
        E_cells = solve_poisson_periodic(mesh, phys, rho, N_cells)
        Phi_cells = solve_poisson_potential_periodic(mesh, phys, rho, N_cells)

        ent = entropy_norm(mesh, phys, f, Phi_cells)
        entropy_hist.append(ent)
        mass_hist.append(np.sum(rho * mesh.hx))

        dt = estimate_dt(mesh, phys, E_cells, cfl=0.3)
        f, E_cells, rho = ssp_rk3_step(mesh, phys, f, N_cells, dt)
        t += dt
        t_hist.append(t)

        if step % 10 == 0 or step == NSTEPS - 1:
            print(f"step {step:3d}  t={t:.4f}  entropy_norm={ent:.6e}  "
                  f"relative={ent/entropy_hist[0]:.6f}")

    t_hist = np.array(t_hist[:-1])  # entropy recorded at start of each step
    entropy_hist = np.array(entropy_hist)
    rel_entropy = entropy_hist / entropy_hist[0]

    # sanity: report whether monotonicity was violated anywhere
    diffs = np.diff(entropy_hist)
    n_increases = np.sum(diffs > 1e-12)
    print(f"\nMonotonicity check: {n_increases} / {len(diffs)} steps where "
          f"entropy_norm increased (should be 0 per Theorem 2.1).")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    axs[0].plot(t_hist, entropy_hist, 'o-', color='crimson')
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$\int f_h^2\, e^{H(x,p,t)}\, p^2\, dp\, d\mu\, dx$")
    axs[0].set_title("Entropy norm  (Theorem 2.1)")

    axs[1].plot(t_hist, rel_entropy, 'o-', color='navy')
    axs[1].axhline(1.0, color='gray', ls='--', lw=1)
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$E(t)/E(0)$")
    axs[1].set_title("Relative entropy  $E(t)/E(0)$")

    plt.tight_layout()
    outpath = "/mnt/user-data/outputs/entropy_norm_plot.png"
    plt.savefig(outpath, dpi=130)
    print(f"\nSaved figure to {outpath}")


if __name__ == "__main__":
    main()
