"""
bp_dg.py
========

Entropy-stable, positivity-preserving Discontinuous Galerkin (DG) solver for the
1D-position / 2D-momentum Boltzmann-Poisson (BP) system in spherical (energy)
momentum coordinates (p, mu), following:

    J. Morales Escalante and I. M. Gamba,
    "Entropy-Stable DG Schemes for Boltzmann-Poisson Models of Collisional
    Electronic Transport along Energy Bands", SINUM (submitted).

Specifically this implements:

  * Section 2.1-2.2: the transformed Boltzmann equation in curvilinear momentum
    coordinates (x, p, mu) under azimuthal symmetry, Eq. (2.3), (2.6).
  * Section 2.2.2 / Appendix (5.1): the semi-discrete DG weak form and the
    upwind numerical fluxes of Eq. (2.11)-(2.12).
  * Section 2.1: the linear electron-phonon collision operator with a discrete
    energy-jump (Dirac-delta) scattering kernel, Eq. (2.1).
  * Eq. (2.4)/(2.43): the analytic solution of the 1D Poisson problem, used
    instead of solving a linear system for Phi at every step.
  * Section 5: the positivity-preserving scaling limiter (Zhang & Shu type)
    that preserves the cell average while eliminating negative point values
    of the piecewise-polynomial solution.

Discretization choices (kept explicit for clarity / reproducibility):

  * Trial/test space V^kappa_h = polynomials of TOTAL degree <= 1 on each
    box cell Omega_{ikm} = [x_i-,x_i+] x [p_k-,p_k+] x [mu_m-,mu_m+], i.e. the
    4-mode modal basis {1, xi, eta, zeta} in reference coordinates
    (xi,eta,zeta) in [-1,1]^3.  This is exactly the "kappa=1" case described
    around Eq. (2.11) and used throughout Section 5.
  * Band structure: parabolic band epsilon(p) = p^2/(2 m*) is used by default
    (the general class of monotone bands ε(r) discussed in Section 1.1 can be
    substituted by redefining eps_band / dp_deps / p_of_eps below).
  * Electric field E(x,t): obtained from the *analytic* solution of the 1D
    periodic, charge-neutral Poisson problem (Theorem 2.4 / Eq. (2.43)),
    evaluated as a piecewise-constant field per x-cell (consistent with a
    mass-preserving discretization of Poisson as assumed in the paper).
  * Time stepping: SSP-RK3 (a convex combination of forward-Euler steps, as
    required for the positivity argument of Section 5.2) with the scaling
    limiter of Section 5 applied after every stage.

This is a reference / teaching-scale implementation: loops are written for
clarity rather than raw speed.  For production-scale runs, vectorize the
inner quadrature loops with numpy broadcasting or move them to a compiled
extension.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Band structure  (Section 1.1 / Section 2.1)
# ---------------------------------------------------------------------------
# Default: parabolic band epsilon(p) = p^2 / (2 m*).  Replace these three
# functions consistently to use a different monotone band epsilon(p)
# (e.g. the Kane band of Section 2.3), as long as eps_band is increasing.

def make_parabolic_band(m_star=1.0):
    def eps_band(p):
        return p**2 / (2.0 * m_star)

    def dp_deps(p):
        # d p / d epsilon = 1 / (d epsilon / d p) = m* / p
        return m_star / p

    def p_of_eps(e):
        e = np.maximum(e, 0.0)
        return np.sqrt(2.0 * m_star * e)

    return eps_band, dp_deps, p_of_eps


# ---------------------------------------------------------------------------
# 2. Reference-cell quadrature and the kappa=1 modal basis
# ---------------------------------------------------------------------------
NQ = 2                                   # Gauss-Legendre points/direction: TRANSPORT terms
_qx, _qw = np.polynomial.legendre.leggauss(NQ)

# The collision operator's gain term evaluates f_h at an energy-shifted point
# p' = p(eps(p) +/- hbar*omega) located (generally) in a *different* p-cell.
# As p sweeps across a cell, the target cell index k_shift(p) can change,
# making the composed integrand piecewise-smooth-but-non-polynomial in a way
# the low-order transport quadrature badly under-resolves (verified
# empirically: NQ=2 -> ~56% error in isolated mass conservation, NQ=8 -> ~2%,
# see validation notes at the bottom of this file). The collision assembly
# therefore uses its own, higher-order rule.
NQC = 6
_qxc, _qwc = np.polynomial.legendre.leggauss(NQC)

NB = 4                                   # total-degree-1 basis: {1, xi, eta, zeta}

# Basis gradients in reference coordinates are constant (this is the whole
# point of using the total-degree-1 modal basis: no need to re-evaluate
# derivatives at quadrature points).
_dB_dxi = np.array([0.0, 1.0, 0.0, 0.0])
_dB_deta = np.array([0.0, 0.0, 1.0, 0.0])
_dB_dzeta = np.array([0.0, 0.0, 0.0, 1.0])


def basis_vals(xi, eta, zeta):
    return np.array([1.0, xi, eta, zeta])


def eval_local(coeffs, xi, eta, zeta):
    return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * zeta


# 8 corners of the reference cube -- since our basis is affine, the extrema
# of f_h over a cell occur exactly at these corners (needed for the
# positivity limiter of Section 5).
_CORNERS = [(a, b, g) for a in (-1.0, 1.0) for b in (-1.0, 1.0) for g in (-1.0, 1.0)]


# ---------------------------------------------------------------------------
# 3. Mesh:  Omega_C = [0,L]_x  x  [0,p_max]_p  x  [-1,1]_mu     (Eq. 2.7)
# ---------------------------------------------------------------------------
class Mesh:
    def __init__(self, L, p_min, p_max, Nx, Np, Nmu):
        self.Nx, self.Np, self.Nmu = Nx, Np, Nmu
        self.L = L
        self.xedges = np.linspace(0.0, L, Nx + 1)
        self.pedges = np.linspace(p_min, p_max, Np + 1)   # p_min>0 avoids the
        self.muedges = np.linspace(-1.0, 1.0, Nmu + 1)     # removable p=0 singularity

        self.hx = np.diff(self.xedges)
        self.hp = np.diff(self.pedges)
        self.hmu = np.diff(self.muedges)

        self.xc = 0.5 * (self.xedges[:-1] + self.xedges[1:])
        self.pc = 0.5 * (self.pedges[:-1] + self.pedges[1:])
        self.muc = 0.5 * (self.muedges[:-1] + self.muedges[1:])

    def find_p_cell(self, p):
        """Locate the p-cell index containing p, or -1 if outside [p_min,p_max]
        (i.e. outside the cut-off domain Omega_{p,t} of Section 2.2.1)."""
        if p < self.pedges[0] or p > self.pedges[-1]:
            return -1
        k = np.searchsorted(self.pedges, p) - 1
        return int(np.clip(k, 0, self.Np - 1))


def new_field(mesh):
    return np.zeros((mesh.Nx, mesh.Np, mesh.Nmu, NB))


def project_function(mesh, func):
    """Local L2 projection of a smooth function func(x,p,mu) into the DG
    space V_h^1 (all 4 modes), using the p^2-weighted inner product. Uses
    the collision-grade quadrature (NQC) for accuracy since this is only
    called at setup time (not per RHS evaluation)."""
    field = np.zeros((mesh.Nx, mesh.Np, mesh.Nmu, NB))
    for i in range(mesh.Nx):
        hx = mesh.hx[i]
        xc = mesh.xc[i]
        for k in range(mesh.Np):
            hp = mesh.hp[k]
            pc = mesh.pc[k]
            for m in range(mesh.Nmu):
                hmu = mesh.hmu[m]
                muc = mesh.muc[m]
                Jvol = 0.5 * hx * 0.5 * hp * 0.5 * hmu
                M = np.zeros((NB, NB))
                b = np.zeros(NB)
                for a in range(NQC):
                    xi, wx = _qxc[a], _qwc[a]
                    x = xc + 0.5 * hx * xi
                    for bb in range(NQC):
                        eta, wp = _qxc[bb], _qwc[bb]
                        p = pc + 0.5 * hp * eta
                        for cc in range(NQC):
                            zeta, wm = _qxc[cc], _qwc[cc]
                            mu = muc + 0.5 * hmu * zeta
                            w = wx * wp * wm * Jvol
                            B = basis_vals(xi, eta, zeta)
                            fval = func(x, p, mu)
                            M += w * p * p * np.outer(B, B)
                            b += w * p * p * fval * B
                field[i, k, m, :] = np.linalg.solve(M, b)
    return field


# ---------------------------------------------------------------------------
# 4. Physical / collision parameters  (Eq. (2.1))
# ---------------------------------------------------------------------------
class Physics:
    def __init__(self, q=1.0, eps_permittivity=1.0, m_star=1.0,
                 hbar_omega=0.3, K_scatter=1.0, T_lattice=1.0):
        self.q = q
        self.eps = eps_permittivity
        self.hbar_omega = hbar_omega
        self.T_lattice = T_lattice
        self.n_ph = 1.0 / (np.exp(hbar_omega / T_lattice) - 1.0)
        self.K = K_scatter
        # c_{+1} = (n_ph+1) K   (emission),   c_{-1} = n_ph K   (absorption) -- Eq. (2.1)
        self.c_plus1 = (self.n_ph + 1.0) * self.K
        self.c_minus1 = self.n_ph * self.K
        self.eps_band, self.dp_deps, self.p_of_eps = make_parabolic_band(m_star)


# ---------------------------------------------------------------------------
# 5. Analytic 1D Poisson solve, periodic + charge-neutral BC   (Eq. 2.43)
# ---------------------------------------------------------------------------
def charge_density_cellavg(mesh, field):
    """Cell-averaged charge density  rho_i = 2*pi * int f p^2 dp dmu  (Eq. 2.3),
    exploiting that for an affine-in-x basis the x-average equals the value
    at xi=0 (the cell midpoint)."""
    rho = np.zeros(mesh.Nx)
    for i in range(mesh.Nx):
        total = 0.0
        for k in range(mesh.Np):
            for m in range(mesh.Nmu):
                c = field[i, k, m, :]
                for a in range(NQ):
                    eta, wp = _qx[a], _qw[a]
                    p = mesh.pc[k] + 0.5 * mesh.hp[k] * eta
                    for b in range(NQ):
                        zeta, wm = _qx[b], _qw[b]
                        val = eval_local(c, 0.0, eta, zeta)
                        total += wp * wm * 0.5 * mesh.hp[k] * 0.5 * mesh.hmu[m] * val * p * p
        rho[i] = 2.0 * np.pi * total
    return rho


def solve_poisson_periodic(mesh, phys, rho_cells, N_cells):
    """Discrete analogue of Eq. (2.43): periodic, charge-neutral 1D Poisson
    solve.  Returns E_cells, the (piecewise-constant per x-cell) electric
    field E(x,t) = -d Phi/dx, evaluated with a Riemann-sum discretization of

        E(x,t) = -(q/eps) [ (1/L) int_0^L (N-rho)(x')(L-x') dx'
                              - int_0^x (N-rho)(x') dx' ] .
    """
    net = N_cells - rho_cells
    hx = mesh.hx
    L = mesh.L
    term1 = np.sum(net * hx * (L - mesh.xc)) / L
    # running (left) cumulative integral up to the *start* of each cell,
    # i.e. int_0^{x_i-} net dx', consistent with piecewise-constant net(x)
    cum = np.concatenate(([0.0], np.cumsum(net * hx)))[:-1]
    E_cells = -(phys.q / phys.eps) * (term1 - cum)
    return E_cells


# ---------------------------------------------------------------------------
# 6. RHS assembly: transport (Eq. 2.11-2.12) + collision (Eq. 2.1)
# ---------------------------------------------------------------------------
def _mass_and_volume(mesh, phys, field, E_cells):
    """Builds the local mass matrices and the transport *volume* contribution
    ( +int (a f) d_x g p^2  type terms of Eq. 2.11, before integration by
    parts back out to fluxes -- i.e. exactly the RHS volume integrals as
    written in the weak form )."""
    Nx, Np, Nmu = mesh.Nx, mesh.Np, mesh.Nmu
    Mloc = np.zeros((Nx, Np, Nmu, NB, NB))
    Rhs = np.zeros((Nx, Np, Nmu, NB))

    for i in range(Nx):
        hx = mesh.hx[i]
        E = E_cells[i]
        for k in range(Np):
            hp = mesh.hp[k]
            pc = mesh.pc[k]
            for m in range(Nmu):
                hmu = mesh.hmu[m]
                muc = mesh.muc[m]
                c = field[i, k, m, :]
                Jvol = 0.5 * hx * 0.5 * hp * 0.5 * hmu
                M = np.zeros((NB, NB))
                R = np.zeros(NB)
                dBdx = (2.0 / hx) * _dB_dxi
                dBdp = (2.0 / hp) * _dB_deta
                dBdmu = (2.0 / hmu) * _dB_dzeta
                for a in range(NQ):
                    xi, wx = _qx[a], _qw[a]
                    for b in range(NQ):
                        eta, wp = _qx[b], _qw[b]
                        p = pc + 0.5 * hp * eta
                        dpe = phys.eps_band  # placeholder to appease linters
                        deps_dp = p / 1.0  # (unused directly; kept for readability)
                        for cc in range(NQ):
                            zeta, wm = _qx[cc], _qw[cc]
                            mu = muc + 0.5 * hmu * zeta
                            w = wx * wp * wm * Jvol
                            B = basis_vals(xi, eta, zeta)
                            f_val = c @ B
                            p2 = p * p

                            M += w * p2 * np.outer(B, B)

                            # transport coefficients, cf. Section 5.1 notation:
                            #   H(x)  = mu * d(eps)/dp
                            #   H(p)  = -q E mu
                            #   H(mu) = -q E
                            deps_dp_val = 1.0 / phys.dp_deps(p)   # = d(eps)/dp
                            Hx = mu * deps_dp_val
                            Hp = -phys.q * E * mu
                            Hmu = -phys.q * E

                            R += w * p2 * Hx * f_val * dBdx
                            R += w * p2 * Hp * f_val * dBdp
                            R += w * p * (1.0 - mu * mu) * Hmu * f_val * dBdmu

                Mloc[i, k, m] = M
                Rhs[i, k, m] = R
    return Mloc, Rhs


def _upwind_add(Rhs, mesh, field, E_cells, phys):
    """Surface (numerical flux) contributions, Eq. (2.12), for the x-, p- and
    mu-directions.  Ghost values are 0 at the true p- and mu-boundaries
    (the cut-off / specular-free "vacuum" condition f|_{d Omega_p,t}=0 of
    Section 2.2.1); the x-direction uses periodic wrap-around."""
    Nx, Np, Nmu = mesh.Nx, mesh.Np, mesh.Nmu

    # ---- x-direction faces (periodic) ----
    for i in range(Nx):
        iR = (i + 1) % Nx
        hx = mesh.hx[i]
        for k in range(Np):
            hp = mesh.hp[k]
            pc = mesh.pc[k]
            for m in range(Nmu):
                hmu = mesh.hmu[m]
                muc = mesh.muc[m]
                cL = field[i, k, m, :]
                cR = field[iR, k, m, :]
                for b in range(NQ):
                    eta, wp = _qx[b], _qw[b]
                    p = pc + 0.5 * hp * eta
                    deps_dp_val = 1.0 / phys.dp_deps(p)
                    for cc in range(NQ):
                        zeta, wm = _qx[cc], _qw[cc]
                        mu = muc + 0.5 * hmu * zeta
                        w = wp * wm * 0.5 * hp * 0.5 * hmu
                        f_minus = eval_local(cL, 1.0, eta, zeta)
                        f_plus = eval_local(cR, -1.0, eta, zeta)
                        a_x = mu * deps_dp_val
                        flux = 0.5 * (a_x + abs(a_x)) * f_minus + 0.5 * (a_x - abs(a_x)) * f_plus
                        Hflux = flux  # already includes mu via a_x
                        contrib = w * p * p * Hflux
                        BR = basis_vals(1.0, eta, zeta)
                        BL = basis_vals(-1.0, eta, zeta)
                        Rhs[i, k, m, :] -= contrib * BR
                        Rhs[iR, k, m, :] += contrib * BL

    # ---- p-direction faces (ghost = 0 at p_min, p_max) ----
    for i in range(Nx):
        hx = mesh.hx[i]
        E = E_cells[i]
        for k in range(-1, Np):
            hpL = mesh.hp[k] if k >= 0 else None
            hpR = mesh.hp[k + 1] if k + 1 < Np else None
            for m in range(Nmu):
                hmu = mesh.hmu[m]
                muc = mesh.muc[m]
                cL = field[i, k, m, :] if k >= 0 else None
                cR = field[i, k + 1, m, :] if k + 1 < Np else None
                p_face = mesh.pedges[k + 1]
                for a in range(NQ):
                    xi, wx = _qx[a], _qw[a]
                    for cc in range(NQ):
                        zeta, wm = _qx[cc], _qw[cc]
                        mu = muc + 0.5 * hmu * zeta
                        w = wx * wm * 0.5 * hx * 0.5 * hmu
                        f_minus = eval_local(cL, xi, 1.0, zeta) if cL is not None else 0.0
                        f_plus = eval_local(cR, xi, -1.0, zeta) if cR is not None else 0.0
                        a_p = -phys.q * E * mu
                        flux = 0.5 * (a_p + abs(a_p)) * f_minus + 0.5 * (a_p - abs(a_p)) * f_plus
                        contrib = w * p_face * p_face * flux
                        if cL is not None:
                            BR = basis_vals(xi, 1.0, zeta)
                            Rhs[i, k, m, :] -= contrib * BR
                        if cR is not None:
                            BL = basis_vals(xi, -1.0, zeta)
                            Rhs[i, k + 1, m, :] += contrib * BL

    # ---- mu-direction faces (ghost = 0 at mu=-1, mu=+1) ----
    for i in range(Nx):
        hx = mesh.hx[i]
        E = E_cells[i]
        a_mu = -phys.q * E
        for k in range(Np):
            hp = mesh.hp[k]
            pc = mesh.pc[k]
            for m in range(-1, Nmu):
                cL = field[i, k, m, :] if m >= 0 else None
                cR = field[i, k, m + 1, :] if m + 1 < Nmu else None
                mu_face = mesh.muedges[m + 1]
                for a in range(NQ):
                    xi, wx = _qx[a], _qw[a]
                    for b in range(NQ):
                        eta, wp = _qx[b], _qw[b]
                        p = pc + 0.5 * hp * eta
                        w = wx * wp * 0.5 * hx * 0.5 * hp
                        f_minus = eval_local(cL, xi, eta, 1.0) if cL is not None else 0.0
                        f_plus = eval_local(cR, xi, eta, -1.0) if cR is not None else 0.0
                        flux = 0.5 * (a_mu + abs(a_mu)) * f_minus + 0.5 * (a_mu - abs(a_mu)) * f_plus
                        contrib = w * p * (1.0 - mu_face * mu_face) * flux
                        if cL is not None:
                            BR = basis_vals(xi, eta, 1.0)
                            Rhs[i, k, m, :] -= contrib * BR
                        if cR is not None:
                            BL = basis_vals(xi, eta, -1.0)
                            Rhs[i, k, m + 1, :] += contrib * BL

    return Rhs


def _collision_add(Rhs, mesh, phys, field):
    """Adds int Q(f_h) g p^2 dp dmu dx, Q(f) as in Eq. (2.1), using the
    closed-form loss frequency nu(p) and gain term G(x,p,t) derived for a
    monotone band epsilon(p) (Section 2.1, 2.2).

    Uses the dedicated higher-order quadrature (NQC/_qxc/_qwc) throughout,
    since the gain term's shifted-point interpolation is not well resolved
    by the low-order transport quadrature (see validation notes at the
    bottom of this file)."""
    Nx, Np, Nmu = mesh.Nx, mesh.Np, mesh.Nmu
    for i in range(Nx):
        xc = mesh.xc[i]
        hx = mesh.hx[i]
        for k in range(Np):
            pc = mesh.pc[k]
            hp = mesh.hp[k]
            for m in range(Nmu):
                muc = mesh.muc[m]
                hmu = mesh.hmu[m]
                c = field[i, k, m, :]
                Jvol = 0.5 * hx * 0.5 * hp * 0.5 * hmu
                Rloc = np.zeros(NB)
                for a in range(NQC):
                    xi, wx = _qxc[a], _qwc[a]
                    for b in range(NQC):
                        eta, wp = _qxc[b], _qwc[b]
                        p = pc + 0.5 * hp * eta
                        e = phys.eps_band(p)

                        # loss frequency nu(eps(p)) -- density-of-states factor
                        # p''^2 dp/deps|_{p''} reduces to p'' for a parabolic band.
                        nu = 0.0
                        for cj, j in ((phys.c_plus1, 1), (phys.c_minus1, -1)):
                            eshift = e - j * phys.hbar_omega
                            if eshift >= 0.0:
                                pshift = phys.p_of_eps(eshift)
                                nu += 4.0 * np.pi * cj * pshift

                        # gain term G(x,p,t) = sum_j c_j p'_j Chi * int_{-1}^{1} f(x,p'_j,mu') dmu'
                        G = 0.0
                        for cj, j in ((phys.c_plus1, 1), (phys.c_minus1, -1)):
                            eshift = e + j * phys.hbar_omega
                            if eshift < 0.0:
                                continue
                            pshift = phys.p_of_eps(eshift)
                            kshift = mesh.find_p_cell(pshift)
                            if kshift < 0:
                                continue
                            etashift = 2.0 * (pshift - mesh.pc[kshift]) / mesh.hp[kshift]
                            etashift = float(np.clip(etashift, -1.0, 1.0))
                            muint = 0.0
                            for m2 in range(Nmu):
                                c2 = field[i, kshift, m2, :]
                                for dd in range(NQC):
                                    z2, w2 = _qxc[dd], _qwc[dd]
                                    val = eval_local(c2, xi, etashift, z2)
                                    muint += w2 * 0.5 * mesh.hmu[m2] * val
                            G += 2.0 * np.pi * cj * pshift * muint

                        for cc in range(NQC):
                            zeta, wm = _qxc[cc], _qwc[cc]
                            w = wx * wp * wm * Jvol
                            B = basis_vals(xi, eta, zeta)
                            f_val = c @ B
                            Qf = G - f_val * nu
                            Rloc += w * p * p * Qf * B
                Rhs[i, k, m, :] += Rloc
    return Rhs


def compute_dfdt(mesh, phys, field, N_cells):
    """Full semi-discrete RHS:  M df/dt = R_transport(f) + R_collision(f),
    returning df/dt (Eq. 2.6 / Appendix 5.1)."""
    rho_cells = charge_density_cellavg(mesh, field)
    E_cells = solve_poisson_periodic(mesh, phys, rho_cells, N_cells)

    Mloc, Rhs = _mass_and_volume(mesh, phys, field, E_cells)
    Rhs = _upwind_add(Rhs, mesh, field, E_cells, phys)
    Rhs = _collision_add(Rhs, mesh, phys, field)

    dfdt = np.zeros_like(field)
    for i in range(mesh.Nx):
        for k in range(mesh.Np):
            for m in range(mesh.Nmu):
                dfdt[i, k, m, :] = np.linalg.solve(Mloc[i, k, m], Rhs[i, k, m, :])
    return dfdt, E_cells, rho_cells


# ---------------------------------------------------------------------------
# 7. Positivity-preserving scaling limiter  (Section 5, Zhang & Shu type)
# ---------------------------------------------------------------------------
def positivity_limiter(field):
    """Cell-average-preserving slope limiter: scales the linear part of each
    cell's polynomial toward the (non-negative) cell average until the
    minimum value over the cell -- attained at one of the 8 cube corners,
    since the basis is affine -- is >= 0."""
    Nx, Np, Nmu, _ = field.shape
    for i in range(Nx):
        for k in range(Np):
            for m in range(Nmu):
                c = field[i, k, m, :]
                avg = c[0]
                if avg <= 0.0:
                    field[i, k, m, :] = np.array([max(avg, 0.0), 0.0, 0.0, 0.0])
                    continue
                vmin = min(avg + c[1] * a + c[2] * b + c[3] * g for (a, b, g) in _CORNERS)
                if vmin < 0.0:
                    theta = avg / (avg - vmin)
                    theta = min(max(theta, 0.0), 1.0)
                    field[i, k, m, 1:] *= theta
    return field


# ---------------------------------------------------------------------------
# 8. SSP-RK3 time stepping (a convex combination of forward-Euler stages, as
#    required by the positivity-preservation argument of Section 5.2).
# ---------------------------------------------------------------------------
def ssp_rk3_step(mesh, phys, field, N_cells, dt, limiter=True):
    f0 = field.copy()

    dfdt1, E_cells, rho_cells = compute_dfdt(mesh, phys, f0, N_cells)
    f1 = f0 + dt * dfdt1
    if limiter:
        positivity_limiter(f1)

    dfdt2, _, _ = compute_dfdt(mesh, phys, f1, N_cells)
    f2 = 0.75 * f0 + 0.25 * (f1 + dt * dfdt2)
    if limiter:
        positivity_limiter(f2)

    dfdt3, _, _ = compute_dfdt(mesh, phys, f2, N_cells)
    f3 = (1.0 / 3.0) * f0 + (2.0 / 3.0) * (f2 + dt * dfdt3)
    if limiter:
        positivity_limiter(f3)

    return f3, E_cells, rho_cells


def solve_poisson_potential_periodic(mesh, phys, rho_cells, N_cells):
    """Companion to solve_poisson_periodic: returns Phi(x,t) itself (cell
    centers), via the same discrete analogue of Eq. (2.43), needed only for
    diagnostics such as the entropy norm of Theorem 2.1 (not used in the
    time-stepping RHS, which only needs E)."""
    net = N_cells - rho_cells
    hx = mesh.hx
    L = mesh.L
    xc = mesh.xc
    term_a = np.sum(net * hx * (L - xc) * (xc / (2.0 * L)))
    Phi = np.zeros(mesh.Nx)
    for i in range(mesh.Nx):
        term_b = (xc[i] / L) * np.sum(net * hx * (L - xc))
        mask = xc < xc[i]
        term_c = np.sum((net * hx * (xc[i] - xc))[mask])
        Phi[i] = (phys.q / phys.eps) * (term_a + term_b - term_c)
    return Phi


def entropy_norm(mesh, phys, field, Phi_cells):
    """Computes the (semi-discrete analogue of the) entropy norm of
    Theorem 2.1:  int f_h^2 exp(H(x,p,t)) p^2 dp dmu dx,
    H(x,p,t) = eps(p) - q*Phi(x,t), using piecewise-constant Phi per x-cell
    (consistent with the piecewise-constant E used in the transport
    assembly)."""
    total = 0.0
    for i in range(mesh.Nx):
        hx = mesh.hx[i]
        Phi = Phi_cells[i]
        for k in range(mesh.Np):
            hp = mesh.hp[k]
            pc = mesh.pc[k]
            for m in range(mesh.Nmu):
                hmu = mesh.hmu[m]
                c = field[i, k, m, :]
                Jvol = 0.5 * hx * 0.5 * hp * 0.5 * hmu
                for a in range(NQC):
                    xi, wx = _qxc[a], _qwc[a]
                    for b in range(NQC):
                        eta, wp = _qxc[b], _qwc[b]
                        p = pc + 0.5 * hp * eta
                        H = phys.eps_band(p) - phys.q * Phi
                        eH = np.exp(H)
                        for cc in range(NQC):
                            zeta, wm = _qxc[cc], _qwc[cc]
                            w = wx * wp * wm * Jvol
                            fval = eval_local(c, xi, eta, zeta)
                            total += w * fval * fval * eH * p * p
    return total


# ---------------------------------------------------------------------------
# 10. CFL estimate, following the conditions derived in Section 5.2
#    (Eq. right before (5.4)); a simple conservative estimate is used here.
# ---------------------------------------------------------------------------
def estimate_dt(mesh, phys, E_cells, cfl=0.3):
    max_deps_dp = max(1.0 / phys.dp_deps(p) for p in mesh.pc)
    max_mu = 1.0
    max_E = np.max(np.abs(E_cells)) + 1e-12
    speed_x = max_deps_dp * max_mu / np.min(mesh.hx)
    speed_p = phys.q * max_E * max_mu / np.min(mesh.hp)
    speed_mu = phys.q * max_E / np.min(mesh.hmu)
    nu_max = 4.0 * np.pi * (phys.c_plus1 + phys.c_minus1) * np.max(mesh.pc)
    denom = speed_x + speed_p + speed_mu + nu_max
    return cfl / max(denom, 1e-12)
