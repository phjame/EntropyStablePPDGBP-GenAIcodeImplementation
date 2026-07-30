"""
run_diode_demo.py
==================

Demo driver for bp_dg.py: simulates a 1D silicon-diode-like problem (Section
2.1 of the paper) with the entropy-stable, positivity-preserving RKDG scheme,
and produces diagnostic plots (charge density, electric field, current, and
positivity/mass-conservation history).

Usage:
    python run_diode_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bp_dg import (
    Mesh, Physics, new_field, project_function, ssp_rk3_step, estimate_dt,
    charge_density_cellavg, solve_poisson_periodic, eval_local,
)


def main():
    # ------------------------------------------------------------------
    # Mesh & physics.  Grid sizes are kept modest (demo scale); increase
    # Nx, Np, Nmu, NSTEPS for production runs (the RHS assembly cost scales
    # roughly like Nx*Np*Nmu*Nmu, dominated by the collision term).
    # ------------------------------------------------------------------
    mesh = Mesh(L=1.0, p_min=0.05, p_max=2.5, Nx=8, Np=8, Nmu=4)
    phys = Physics(q=1.0, eps_permittivity=1.0, m_star=1.0,
                    hbar_omega=0.3, K_scatter=1.0, T_lattice=1.0)

    field = new_field(mesh)

    # Initial condition: a normalized, isotropic Maxwellian-like bump in p,
    # uniform in x. Uses a proper local L2 projection (all 4 DG modes) rather
    # than a crude cell-average-only assignment -- important for the
    # collision term's shifted-point interpolation accuracy (see validation
    # notes at the bottom of bp_dg.py).
    p0, sigma = 0.9, 0.35

    def f0(x, p, mu):
        return np.exp(-((p - p0) ** 2) / (2 * sigma ** 2))

    field = project_function(mesh, f0)

    # normalize total mass to a reference value (scale ALL modes uniformly --
    # scaling mode 0 alone would be inconsistent with the projected slopes)
    rho0 = charge_density_cellavg(mesh, field)
    total_mass0 = np.sum(rho0 * mesh.hx)
    field *= 0.4 / total_mass0

    # Doping background N(x): a simple n+ - n - n+ diode profile
    N_cells = 0.2 + 0.3 * (mesh.xc < 0.2) + 0.3 * (mesh.xc > 0.8)

    NSTEPS = 30
    mass_hist = []
    minval_hist = []
    t = 0.0
    for step in range(NSTEPS):
        rho = charge_density_cellavg(mesh, field)
        E_cells = solve_poisson_periodic(mesh, phys, rho, N_cells)
        dt = estimate_dt(mesh, phys, E_cells, cfl=0.3)
        field, E_cells, rho = ssp_rk3_step(mesh, phys, field, N_cells, dt)
        t += dt

        mass_hist.append(np.sum(rho * mesh.hx))
        minval_hist.append(field[..., 0].min())

        if step % 10 == 0 or step == NSTEPS - 1:
            print(f"step {step:3d}  t={t:.4f}  dt={dt:.2e}  "
                  f"mass={mass_hist[-1]:.6f}  min(cell avg f)={minval_hist[-1]:.3e}")

    # ------------------------------------------------------------------
    # Diagnostics / plots
    # ------------------------------------------------------------------
    rho_final = charge_density_cellavg(mesh, field)
    E_final = solve_poisson_periodic(mesh, phys, rho_final, N_cells)

    # current density J(x) = 2*pi * int v(p) f p^2 dp dmu, v(p)=mu*deps/dp
    J = np.zeros(mesh.Nx)
    for i in range(mesh.Nx):
        tot = 0.0
        for k in range(mesh.Np):
            p = mesh.pc[k]
            deps_dp = 1.0 / phys.dp_deps(p)
            for m in range(mesh.Nmu):
                c = field[i, k, m, :]
                val = eval_local(c, 0.0, 0.0, 0.0)  # x-cell-average (affine cancels)
                mu = mesh.muc[m]
                tot += val * mu * deps_dp * p * p * mesh.hp[k] * mesh.hmu[m]
        J[i] = 2 * np.pi * tot

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].plot(mesh.xc, rho_final, 'o-')
    axs[0, 0].set_title("Charge density rho(x)")
    axs[0, 0].set_xlabel("x")

    axs[0, 1].plot(mesh.xc, E_final, 'o-', color='darkorange')
    axs[0, 1].set_title("Electric field E(x)  (analytic Poisson, Eq. 2.4/2.43)")
    axs[0, 1].set_xlabel("x")

    axs[0, 2].plot(mesh.xc, J, 'o-', color='green')
    axs[0, 2].set_title("Current density J(x)")
    axs[0, 2].set_xlabel("x")

    axs[1, 0].plot(mass_hist)
    axs[1, 0].set_title("Total mass vs. time step (conservation check)")
    axs[1, 0].set_xlabel("step")

    axs[1, 1].plot(minval_hist)
    axs[1, 1].axhline(0, color='r', ls='--')
    axs[1, 1].set_title("min(cell-average f) vs. step (positivity check)")
    axs[1, 1].set_xlabel("step")

    # pdf slice f(x_mid, p, mu=0) at final time
    imid = mesh.Nx // 2
    F = np.zeros((mesh.Np,))
    for k in range(mesh.Np):
        # find mu-cell containing mu=0
        m0 = mesh.find_p_cell.__self__.Nmu // 2 if False else np.searchsorted(mesh.muedges, 0.0) - 1
        m0 = int(np.clip(m0, 0, mesh.Nmu - 1))
        c = field[imid, k, m0, :]
        F[k] = eval_local(c, 0.0, 0.0, 0.0)
    axs[1, 2].plot(mesh.pc, F, 'o-', color='purple')
    axs[1, 2].set_title(f"f(x={mesh.xc[imid]:.2f}, p, mu~0) at final t")
    axs[1, 2].set_xlabel("p")

    plt.tight_layout()
    outpath = "./bp_dg_diode_demo.png"
    plt.savefig(outpath, dpi=130)
    print(f"\nSaved figure to {outpath}")


if __name__ == "__main__":
    main()
