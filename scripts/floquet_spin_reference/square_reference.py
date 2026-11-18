#!/usr/bin/env python3
"""Canonical spin block of the half-filled, open four-edge Hubbard plaquette.

The expansion is an independent Fock-matrix implementation, not an oracle
by itself. The exact static comparison diagonalizes the full 70-state H(t)
and aligns its low-energy frame to the site-spin basis by a polar factor; it
neither fits perturbative coefficients nor uses them to select the frame.
The driven comparison solves the full time-domain Schrödinger equation, then
aligns periodic Floquet modes in the zero-photon (not stroboscopic) gauge.
Use --self-check (NumPy/SciPy), or --self-check --static-only (NumPy only).
Set OPENBLAS_NUM_THREADS=1 for these small dense calculations.
"""
import argparse
import hashlib
from itertools import combinations
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

U, W = 4.0, 5.0
AMPLITUDE = .4
ED_T = (.16, .08, .04)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
DIAGONALS = ((0, 2), (1, 3))
NORM = np.linalg.norm


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fock():
    # Site-major modes (up0, down0, ...), ascending creators. In this basis
    # the site-tensor spin embedding needs no fermionic reordering sign.
    basis = [s for s in range(256) if bin(s).count('1') == 4]
    lookup = {s: i for i, s in enumerate(basis)}
    nd = np.array([sum((s >> (2*i) & 3) == 3 for i in range(4)) for s in basis])
    p = np.zeros((len(basis), 16))
    for a in range(16):
        state = sum(1 << (2*i + (a >> (3-i) & 1)) for i in range(4))
        p[lookup[state], a] = 1
    require(np.array_equal(np.bincount(nd), [16, 48, 6]), 'incorrect Nd sectors')
    require(np.array_equal(np.any(p, axis=1), nd == 0) and
            np.array_equal(p.T @ p, np.eye(16)), 'incorrect P embedding')

    directed = np.zeros((4, len(basis), len(basis)))
    for op, (i, j) in zip(directed, EDGES):
        for col, state in enumerate(basis):
            for spin in (0, 1):
                dst, src = 2*i+spin, 2*j+spin
                if state >> src & 1 and not state >> dst & 1:
                    removed = state ^ (1 << src)
                    sign = (-1)**(bin(state & ((1 << src)-1)).count('1') +
                                  bin(removed & ((1 << dst)-1)).count('1'))
                    op[lookup[removed | (1 << dst)], col] = sign
    return nd, p, directed


def orders(v, h0, p, nd):
    """Dense PV R V P and H4; split the middle resolvent by doublon count."""
    q = ~np.any(p, axis=1)
    min_abs_q = float(np.min(np.abs(h0[q])))
    require(np.all(np.isfinite(h0)) and min_abs_q > .5, 'resonant Q denominator')
    r = np.zeros_like(h0)
    r[q] = [-1./float(e) for e in h0[q]]
    require(NORM(v - v.conj().T) < 1e-12, 'non-Hermitian hopping')
    vp = v @ p
    require(NORM(p.T @ vp) < 1e-12, 'PVP must vanish')
    rvp = r[:, None]*vp
    h2 = vp.conj().T @ rvp
    metric2 = rvp.conj().T @ rvp
    folded = -.5*(metric2 @ h2 + h2 @ metric2)
    mid = v @ rvp
    h3 = rvp.conj().T @ mid
    require(NORM(h3) < 1e-12, 'odd order must vanish on the bipartite plaquette')
    bare = [mid[nd == n].conj().T @ (r[nd == n, None]*mid[nd == n]) for n in range(3)]
    h4 = sum(bare) + folded
    require(max(NORM(h2-h2.conj().T), NORM(h4-h4.conj().T)) < 1e-12,
            'non-Hermitian canonical expansion')
    return h2, h4, folded, bare, min_abs_q


def spin_operators():
    s = .5*np.array(([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))
    local = [[np.kron(np.kron(np.eye(2**i), op), np.eye(2**(3-i))) for op in s]
             for i in range(4)]
    pair = {ij: sum(x @ y for x, y in zip(local[ij[0]], local[ij[1]]))
            for ij in combinations(range(4), 2)}
    chi = {}
    for i, j, k in combinations(range(4), 3):
        chi[i, j, k] = sum(
            local[i][a] @ (local[j][b] @ local[k][c] - local[j][c] @ local[k][b])
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
        sij, sjk = pair[i, j], pair[j, k]
        require(NORM(chi[i, j, k] - 1j*(sij @ sjk - sjk @ sij)) < 1e-13,
                'wrong chirality orientation')
    return pair, chi


def coefficient(op, h):
    value = np.vdot(op, h)/np.vdot(op, op)
    require(abs(value.imag) < 1e-12, 'nonreal spin projection')
    return float(value.real)


def polar(overlap):
    # overlap = P^dag Psi = L s R^dag, rotate Psi by R L^dag. This only
    # needs the 16x16 overlap, so it also works for time-periodic Floquet modes.
    left, singular, right_h = np.linalg.svd(overlap)
    rotation = right_h.conj().T @ left.conj().T
    positive = overlap @ rotation
    herm = NORM(positive-positive.conj().T)
    require(singular[-1] > .9 and herm < 1e-12 and
            np.linalg.eigvalsh((positive+positive.conj().T)/2)[0] > .9,
            'P-frame polar overlap is not positive/well conditioned')
    return rotation, float(singular[-1]), float(herm)


def static_check(report, nd, p, directed, pair, chi):
    v = sum(directed + directed.transpose(0, 2, 1))
    h0 = U*nd
    h2, h4, folded, bare, _ = orders(v, h0, p, nd)
    eye = np.eye(16)
    sides = [pair[tuple(sorted(e))] for e in EDGES]
    diagonals = [pair[e] for e in DIAGONALS]
    ring = [sides[0] @ sides[2], sides[1] @ sides[3], diagonals[0] @ diagonals[1]]
    j_edges = sum(s-eye/4 for s in sides)
    j_diag = sum(s-eye/4 for s in diagonals)
    # Open single plaquette, not the infinite square lattice's -24 edge term:
    # each edge is on only one plaquette. Constants fix the polarized energy to 0.
    ref2 = 4/U*j_edges
    ref4 = (-20*j_edges + 4*j_diag + 80*(ring[0]+ring[1]-ring[2]-eye/16))/U**3
    gram = [[np.vdot(a, b).real for b in ring] for a in ring]
    ring_coeffs = np.linalg.solve(gram, [np.vdot(a, U**3*h4).real for a in ring])
    chi4 = [coefficient(c, U**3*h4) for c in chi.values()]
    static = report['static'] = dict(
        H2_known_matrix_residual=float(NORM(h2-ref2)),
        H4_known_matrix_residual=float(NORM(h4-ref4)),
        H4_edge_U3=[coefficient(s, U**3*h4) for s in sides],
        H4_diagonal_U3=[coefficient(s, U**3*h4) for s in diagonals],
        H4_ring_U3=ring_coeffs.tolist(), H4_chi_U3=chi4,
        H4_bare_by_Nd=[float(NORM(x)) for x in bare], H4_folded=float(NORM(folded)))
    require(max(static['H2_known_matrix_residual'], static['H4_known_matrix_residual']) < 1e-12,
            'full-matrix static plaquette anchor failed')
    require(np.allclose(ring_coeffs, [80, 80, -80], rtol=0, atol=1e-11) and
            max(map(abs, chi4)) < 1e-12, 'static ring/chirality coefficients failed')

    rows = static['exact_ED'] = []
    hopping_norm = NORM(v, ord=2)
    for t in ED_T:
        # Weyl bound for the whole interval [0,t]: the lowest 16 states
        # remain isolated from the 54 charge states, so no energy-label guess.
        gap_bound = U - 2*t*hopping_norm
        require(gap_bound > 0, 'static P continuation lacks a gap bound')
        h = np.diag(h0) + t*v
        energies, vectors = np.linalg.eigh(h)
        psi = vectors[:, :16]
        rotation, p_min, herm = polar(p.T @ psi)
        frame = psi @ rotation
        block = frame.conj().T @ h @ frame
        residual = float(NORM(h @ frame - frame @ block))
        require(residual < 1e-11 and NORM(frame.conj().T @ frame - eye) < 1e-12,
                'static ED eigen/isometry residual too large')
        delta = block - t**2*ref2
        rem = delta - t**4*ref4
        rows.append(dict(t_ev=t, gap_lower_bound_ev=float(gap_bound),
                         gap_ev=float(energies[16]-energies[15]), P_overlap_min=p_min,
                         P_overlap_antihermitian=herm, eigen_residual_ev=residual,
                         H2_error_ev=float(NORM(delta)), H24_error_ev=float(NORM(rem)),
                         no_folded_error_ev=float(NORM(rem+t**4*folded)),
                         no_Nd2_error_ev=float(NORM(rem+t**4*bare[2]))))
    for key, lo, hi in (('H2_error_ev', 14, 17), ('H24_error_ev', 57, 66),
                        ('no_folded_error_ev', 14, 18), ('no_Nd2_error_ev', 14, 18)):
        ratios = [a[key]/b[key] for a, b in zip(rows, rows[1:])]
        static[key + '_halving_ratios'] = ratios
        require(all(lo < x < hi for x in ratios), 'unexpected static scaling: %s %r' % (key, ratios))
    last = rows[-1]
    require(last['H24_error_ev'] > 100*last['eigen_residual_ev'] and
            min(last['no_folded_error_ev'], last['no_Nd2_error_ev']) > 100*last['H24_error_ev'],
            'retained H4 improvement not resolved by ED')


def peierls(theta, helicity):
    # Sites CCW: (0,0), (1,0), (1,1), (0,1). The listed index edges have
    # d_ij = r_j-r_i = (+x,+y,-x,-y); c_i^dag c_j moves in reverse.
    x, y = np.cos(theta), helicity*np.sin(theta)
    return np.exp(-1j*AMPLITUDE*np.stack((x, y, -x, -y), axis=-1))


def sambe_orders(nd, p, directed, harmonic, helicity, grid):
    theta = 2*np.pi*np.arange(grid)/grid
    q = np.arange(-harmonic, harmonic+1)
    amp = np.exp(1j*q[:, None]*theta) @ peierls(theta, helicity)/grid
    fwd = directed.reshape(4, -1)
    rev = directed.transpose(0, 2, 1).reshape(4, -1)
    modes = (amp @ fwd + amp[::-1].conj() @ rev).reshape(-1, len(nd), len(nd))
    # H(theta)=sum h_q exp(-iq theta), positive-time basis exp(+in theta).
    # At fourth order each closed path stays within |n|<=2*harmonic;
    # this cutoff discards no path of the band-limited drive under test.
    m = 2*harmonic
    photons = np.arange(-m, m+1)
    zero = np.zeros_like(modes[0])
    v = np.block([[modes[j-i+harmonic] if abs(j-i) <= harmonic else zero
                   for j in photons] for i in photons])
    h0 = (W*photons[:, None] + U*nd).ravel()
    p0 = np.zeros((len(h0), p.shape[1]))
    p0[m*len(nd):(m+1)*len(nd)] = p
    return orders(v, h0, p0, np.tile(nd, len(photons)))


def propagate(nd, p, directed, t, helicity, grid, rtol):
    from scipy.integrate import solve_ivp
    from scipy.linalg import schur

    size = len(nd)
    eye = np.eye(size)
    fwd = directed.reshape(4, -1)
    rev = directed.transpose(0, 2, 1).reshape(4, -1)

    def rhs(theta, x):
        phase = peierls(theta, helicity)
        hop = (phase @ fwd + phase.conj() @ rev).reshape(size, size)
        x = x.reshape(size, size)
        return (-1j/W*(U*nd[:, None]*x + t*(hop @ x))).ravel()

    theta = np.linspace(0., 2*np.pi, grid+1)
    solution = solve_ivp(rhs, (theta[0], theta[-1]), eye.astype(complex).ravel(),
                         t_eval=theta, method='DOP853', rtol=rtol, atol=rtol/100)
    require(solution.success, 'forward ODE solver failed: ' + solution.message)
    evolution = solution.y.T.reshape((-1, size, size))
    endpoint = evolution[-1]
    unity = float(NORM(endpoint.conj().T @ endpoint-eye))
    triangular, vectors = schur(endpoint, output='complex')
    lam = triangular.diagonal()
    eigen_res = float(NORM(endpoint @ vectors - vectors*lam))
    require(max(unity, eigen_res) < 1e-10, 'nonunitary propagation or Schur residual')

    # The instantaneous uniform-A hopping is gauge-equivalent to V(0 drive),
    # so Duhamel gives this unitary perturbation radius for all scales [0,t].
    # Normal-matrix spectral disks are disjoint between P and both charge
    # sectors; hence the nearest 16 eigenvalues belong to P by continuity.
    static_hop = sum(directed + directed.transpose(0, 2, 1))
    radius = t*theta[-1]/W*NORM(static_hop, ord=2)
    charge_chord = min(abs(np.exp(-1j*theta[-1]*k*U/W)-1) for k in (1, 2))
    require(2*radius < charge_chord, 'drive P-continuation disks are not disjoint')
    distance = np.abs(lam-1)
    chosen = np.argsort(distance)[:p.shape[1]]
    require(np.count_nonzero(distance < charge_chord/2) == p.shape[1] and
            max(distance[chosen]) <= radius+1e-10, 'incorrect quasienergy subspace')
    eps = -W/theta[-1]*np.angle(lam[chosen])
    # The ν=0 component of u_j(theta)=exp(+i eps_j theta/W) U(theta) psi_j(0)
    # is the P overlap in Sambe space. Averaging it, then taking its polar
    # factor, avoids comparing to a bare t=0 stroboscopic (kick-dependent) gauge.
    sampled = (p.T @ evolution @ vectors[:, chosen])*np.exp(1j*theta[:, None, None]*eps/W)
    closure = float(NORM(sampled[-1]-sampled[0]))
    rotation, p_min, herm = polar(np.mean(sampled[:-1], axis=0))
    require(closure < 1e-10, 'nonperiodic projected Floquet modes')
    block = rotation.conj().T @ (eps[:, None]*rotation)
    return block, dict(unitarity=unity, schur_eigen_residual=eigen_res,
                       spectrum_disk_margin=float(charge_chord-2*radius),
                       P_overlap_min=p_min, P_overlap_antihermitian=herm,
                       P_mode_periodicity=closure)


def driven_check(report, nd, p, directed, chi):
    # Uniform four-edge hopping on this fixture cancels the chi012 signal.
    # Use unequal positive bonds to expose a fourth-order sign discriminator.
    weights = (1., .8, .6, .9)
    directed = directed*np.array(weights)[:, None, None]
    driven = report['driven'] = dict(W_ev=W, amplitude=AMPLITUDE, helicity=1, edge_weights=weights,
                                     orientation='a(theta)=A(cos(theta),sin(theta)); sites CCW')
    h2, h4, _, _, q_gap = sambe_orders(nd, p, directed, 4, 1, 128)
    wide2, wide4, _, _, _ = sambe_orders(nd, p, directed, 6, 1, 256)
    rev2, rev4, _, _, _ = sambe_orders(nd, p, directed, 4, -1, 128)
    chi012 = chi[0, 1, 2]
    chi4 = coefficient(chi012, wide4)
    driven.update(H4_chi012=chi4, wrong_index_H4_chi012=coefficient(chi012, rev4),
                  H2_cutoff_change=float(NORM(h2-wide2)), H4_cutoff_change=float(NORM(h4-wide4)),
                  H2_reversal_change=float(NORM(h2-rev2)), minimum_abs_Q_energy_ev=q_gap)
    rows = driven['exact_propagation'] = []
    t_values = (.1, .05, .025)
    driven.update(sambe_settings=[[4, 8, 128], [6, 12, 256]],
                  sambe_settings_order='harmonic_max, photon_max, DFT_size',
                  propagation_settings=[[128, 2e-12], [256, 2e-13]],
                  propagation_settings_order='quadrature_size, DOP853_rtol')
    for t in t_values:
        coarse, _ = propagate(nd, p, directed, t, 1, 128, 2e-12)
        block, diag = propagate(nd, p, directed, t, 1, 256, 2e-13)
        if t == t_values[0]:
            forward = block
        retained = t**2*wide2+t**4*wide4
        reversed_order = t**2*rev2+t**4*rev4
        rows.append(dict(t_ev=t, H2_error_ev=float(NORM(block-t**2*wide2)),
                         H24_error_ev=float(NORM(block-retained)),
                         wrong_index_H24_error_ev=float(NORM(block-reversed_order)),
                         chi012_ev=coefficient(chi012, block),
                         retained_chi012_ev=t**4*chi4,
                         propagation_refinement_ev=float(NORM(block-coarse)), **diag))
    rev, _ = propagate(nd, p, directed, t_values[0], -1, 256, 2e-13)
    # Circular A(theta+pi)=-A(theta), so reversal is complex conjugation
    # followed by theta->pi-theta. A time shift leaves the zero mode unchanged.
    driven.update(reversed_propagation_chi012_ev=coefficient(chi012, rev),
                  reverse_conjugation_residual_ev=float(NORM(rev-forward.conj())))
    for key, lo, hi in (('H2_error_ev', 14, 18), ('H24_error_ev', 57, 72),
                        ('wrong_index_H24_error_ev', 14, 19)):
        ratios = [a[key]/b[key] for a, b in zip(rows, rows[1:])]
        driven[key+'_halving_ratios'] = ratios
        require(all(lo < x < hi for x in ratios), 'unexpected driven scaling: %s %r' % (key, ratios))
    require(chi4 > .04 and driven['wrong_index_H4_chi012'] < -.04 and
            driven['reversed_propagation_chi012_ev'] < -4e-6 and
            all(.9 < row['chi012_ev']/row['retained_chi012_ev'] < 1.1 for row in rows),
            'physical forward propagation does not confirm the signed fourth-order chi012')
    require(driven['reverse_conjugation_residual_ev'] < 1e-11 and
            driven['H2_reversal_change'] < 1e-12, 'protocol reversal check failed')
    for row in rows:
        t = row['t_ev']
        cutoff_error = t**2*driven['H2_cutoff_change'] + t**4*driven['H4_cutoff_change']
        require(max(cutoff_error, row['propagation_refinement_ev']) < .01*row['H24_error_ev'],
                'harmonic or propagation refinement did not resolve the H24 residual')
        require(row['wrong_index_H24_error_ev'] > 5*row['H24_error_ev'],
                'ineffective reversed-index negative control')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.add_argument('--static-only', action='store_true', help='run the NumPy-only ED checks')
    args = parser.parse_args()
    report = dict(U_ev=U, fock_dimension=70, P_dimension=16, python=platform.python_version(),
                  numpy=np.__version__, warning_options=sys.warnoptions,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        nd, p, directed = fock()
        pair, chi = spin_operators()
        static_check(report, nd, p, directed, pair, chi)
        if not args.static_only:
            import scipy
            report['scipy'] = scipy.__version__
            driven_check(report, nd, p, directed, chi)
    except Exception:
        report.update(exit_code=1, traceback=traceback.format_exc())
    else:
        report['exit_code'] = 0
    print(json.dumps(report, indent=2, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
