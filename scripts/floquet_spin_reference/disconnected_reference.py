#!/usr/bin/env python3
"""Dense Fock check of signed photon-window algebra for two disconnected dimers.

This tests the given canonical fourth-order formula with an *exactly* band-
limited hopping fixture. H2/H4 are coefficients in a formal common hopping
scale, not a finite-hopping convergence claim. This is neither a proof of the
SW formula nor a check of dynamics, chirality, or Peierls-spectrum convergence.

For four closed hops, |n_s| <= min(s, 4-s) H, hence M >= 2H retains every path.
For fixed two-hop paths on A and B, let x,y be the first local virtual energies
and z the net photon energy gained on A (B gains -z). The sum of the four
interleaved inverse denominator products, with the three R-minus signs taken
out, is (1/(y+z) + 1/(x-z))/(x*y): their common middle energy is x+y (Nd=2).
If z != 0, the two sequential paths have middle energies z and -z (Nd=0) and
sum to its negative. If z == 0, they return to P and are excluded; the remaining
1/(x*x*y) + 1/(x*y*y) cancels the cross term from -{PV R^2 V P,H2}/2. The local
hops are even fermion operators, so shuffling does not alter their numerators.
At H=M=1 the folded terms survive, but both middle sectors n_2=+/-2 are
removed, clipping Nd=0 sequential and Nd=2 interleaved paths.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

U, W, H = 4.0, 5.0, 1
# (t in eV, cosine modulation, imaginary sine modulation, phase in radians)
DRIVES = ((0.70, 0.46, 0.23, 0.17), (0.93, -0.37, 0.31, -0.61))
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fock(drives):
    sites = 2 * len(drives)
    count = lambda state: bin(state).count('1')
    # Site-major modes (up0, down0, up1, down1, ...), ascending creators.
    basis = [state for state in range(1 << (2 * sites)) if count(state) == sites]
    index = {state: i for i, state in enumerate(basis)}
    nd = np.array([sum(state >> (2 * site) & 3 == 3 for site in range(sites))
                   for state in basis])
    spin = np.zeros((len(basis), 2**sites))
    for bits in range(2**sites):
        occupied = [2 * site + (bits >> (sites - 1 - site) & 1) for site in range(sites)]
        spin[index[sum(1 << mode for mode in occupied)], bits] = 1  # no permutation
    require(np.array_equal(np.any(spin, axis=1), nd == 0) and
            np.array_equal(spin.T @ spin, np.eye(2**sites)), 'invalid P embedding')

    blocks = {q: np.zeros((len(basis), len(basis)), complex) for q in range(-H, H + 1)}
    forwards = []
    for bond, (t, alpha, beta, phi) in enumerate(drives):
        forward = np.zeros_like(blocks[0])
        for col, state in enumerate(basis):
            for spin_mode in (0, 1):
                dst = 4 * bond + spin_mode
                src = dst + 2
                if state >> src & 1 and not state >> dst & 1:
                    vacant = state ^ (1 << src)
                    sign = (-1)**(count(state & ((1 << src) - 1)) +
                                  count(vacant & ((1 << dst) - 1)))
                    forward[index[vacant | (1 << dst)], col] = sign
        forwards.append(forward)
        # t(theta)=t*(1+alpha*cos(theta-phi)+i*beta*sin(theta-phi)); reverse is t*.
        # The following are exact coefficients of exp(-iq theta), zero for |q|>H.
        coeff = {0: t, **{q: .5*t*(alpha-q*beta)*np.exp(1j*q*phi) for q in (-1, 1)}}
        for q, block in blocks.items():
            block += coeff[q]*forward + coeff[-q].conjugate()*forward.T
    return nd, spin, blocks, forwards


def dense_orders(cell, cutoff):
    nd, spin, blocks, _ = cell
    photons = np.arange(-cutoff, cutoff + 1)
    zero = np.zeros_like(blocks[0])
    # Temporal basis exp(+in theta): bra n, ket m -> V_(m-n); derivative +nW.
    v = np.block([[blocks.get(m-n, zero) for m in photons] for n in photons])
    require(np.linalg.norm(v-v.conj().T) < TOL, 'non-Hermitian Sambe hopping')
    p = np.zeros((len(v), spin.shape[1]))
    p[cutoff*len(nd):(cutoff+1)*len(nd)] = spin
    h0 = (U*nd[None, :] + W*photons[:, None]).ravel()
    q = ~np.any(p, axis=1)
    gap = float(np.min(np.abs(h0[q])))
    require(np.all(np.isfinite(h0[q])) and gap > .5, 'resonant or nonfinite Q denominator')
    r = np.zeros_like(h0)
    # Scalar division works with strict warnings on the host's NumPy 1.20.
    r[q] = [-1.0/float(energy) for energy in h0[q]]
    vp = v @ p
    require(np.linalg.norm(p.T @ vp) < TOL, 'PVP is not zero')
    rvp = r[:, None] * vp
    h2 = vp.conj().T @ rvp
    n2 = rvp.conj().T @ rvp
    folded = -.5*(n2 @ h2 + h2 @ n2)
    mid = v @ rvp
    nd_sambe = np.tile(nd, len(photons))
    n_sambe = np.repeat(photons, len(nd))
    require(np.linalg.norm(mid[nd_sambe == 1]) < TOL, 'two-hop Nd=1 should be unreachable')

    def bare(mask):
        return mid[mask].conj().T @ (r[mask, None]*mid[mask])

    signed_bare = {int(n): bare(n_sambe == n) for n in photons}
    h4 = sum(signed_bare.values()) + folded
    require(np.linalg.norm(h4-h4.conj().T) < TOL, 'non-Hermitian H4')
    size = len(nd)
    time_block = v[(cutoff+1)*size:(cutoff+2)*size, cutoff*size:(cutoff+1)*size]
    return h2, h4, folded, signed_bare, bare(nd_sambe == 2), gap, time_block


def self_check(report):
    ab, a, b = (fock(ds) for ds in (DRIVES, DRIVES[:1], DRIVES[1:]))
    require([len(cell[0]) for cell in (ab, a, b)] == [70, 6, 6], 'wrong half-filled dimensions')
    eye = np.eye(4)

    def defect(ab_term, a_term, b_term):
        return ab_term - np.kron(a_term, eye) - np.kron(eye, b_term)

    norm = np.linalg.norm
    windows = report['windows'] = {}
    records = {}
    for m in (1, 2, 3):
        ab_res, a_res, b_res = (dense_orders(cell, m) for cell in (ab, a, b))
        h2 = defect(ab_res[0], a_res[0], b_res[0])
        h4 = defect(ab_res[1], a_res[1], b_res[1])
        folded = defect(ab_res[2], a_res[2], b_res[2])
        bare = {n: defect(ab_res[3][n], a_res[3][n], b_res[3][n]) for n in ab_res[3]}
        nd2 = ab_res[4]  # impossible on either separate dimer
        records[m] = (h4, folded, bare, nd2, ab_res[6])
        windows[m] = dict(photon_range=[-m, m], minimum_abs_Q_energy_ev=min(
            ab_res[5], a_res[5], b_res[5]), H2_defect_ev=norm(h2), H4_defect_ev=norm(h4),
            folded_defect_ev=norm(folded), bare_Nd2_ev=norm(nd2),
            bare_Nd0_defect_ev=norm(sum(bare.values())-nd2),
            defect_without_Nd2_ev=norm(h4-nd2), defect_without_folded_ev=norm(h4-folded))
        require(norm(h2) < TOL, 'H2 is not additive')

    tight, tight_fold, tight_bare, _, _ = records[1]
    full, full_fold, full_bare, nd2, time_block = records[2]
    wider, _, _, _, _ = records[3]
    missing = full_bare[-2] + full_bare[2]
    shared_bare_change = max(norm(tight_bare[n]-full_bare[n]) for n in tight_bare)
    singlet = np.array([0., 1., -1., 0.])
    pair = .5*np.outer(singlet, singlet)
    pair_pair = np.kron(pair, pair)
    singlet_weight = np.vdot(pair_pair, tight)
    minus_weight = float(np.vdot(pair_pair, full_bare[-2]).real)
    plus_weight = float(np.vdot(pair_pair, full_bare[2]).real)
    report['window_controls'] = dict(missing_n_minus_2_ev=norm(full_bare[-2]),
        missing_n_plus_2_ev=norm(full_bare[2]), missing_n_minus_2_singlet_pair_ev=minus_weight,
        missing_n_plus_2_singlet_pair_ev=plus_weight, shared_bare_change_ev=shared_bare_change,
        difference_plus_missing_ev=norm(tight-full+missing),
        folded_change_ev=norm(tight_fold-full_fold), full_to_wider_ev=norm(full-wider),
        tight_singlet_pair_weight_ev=float(singlet_weight.real),
        tight_nonsinglet_residual_ev=norm(tight-singlet_weight*pair_pair))

    # Independently time-project the (n=1,m=0) block from H(theta); no propagation.
    theta = 2*np.pi*np.arange(16)/16
    sampled = np.zeros((len(theta), *time_block.shape), complex)
    for (t, alpha, beta, phi), forward in zip(DRIVES, ab[3]):
        amp = t*(1+alpha*np.cos(theta-phi)+1j*beta*np.sin(theta-phi))
        sampled += amp[:, None, None]*forward + amp.conj()[:, None, None]*forward.T
    bra = np.exp(1j*theta).conj()[:, None, None]
    projection = norm(time_block-np.mean(bra*sampled, axis=0))
    reversed_projection = norm(time_block-np.mean(bra.conj()*sampled, axis=0))
    report['time_projection'] = dict(physical_residual_ev=projection,
                                     reversed_index_residual_ev=reversed_projection)
    require(projection < TOL and reversed_projection > .1, 'time projection or sign control failed')
    # The directed time projection checks h_(m-n); these asymmetric signed
    # fixture values independently reject negating nW while retaining h_(m-n).
    require(minus_weight > 1e-6 and plus_weight < -1e-6, 'signed boundary weights or nW sign failed')
    require(norm(tight) > 1e-4, 'tight-window counterexample is ineffective')
    require(max(norm(tight-full+missing), norm(tight_fold-full_fold), shared_bare_change) < TOL,
            'change is not confined to the signed bare n_2=+/-2 paths')
    require(norm(tight-singlet_weight*pair_pair) < TOL, 'unexpected tight-window spin structure')
    # Report all measured terms before asserting cancellation; never hide a full-window defect.
    require(norm(full) < TOL and norm(wider) < TOL, 'unexpected full-window nonadditivity')
    require(norm(full-nd2) > 1e-4 and norm(full-full_fold) > 1e-4,
            'Nd=2 or folded negative control is ineffective')
    require(windows[2]['bare_Nd0_defect_ev'] > 1e-4, 'Nd=0 sequential paths are ineffective')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    report = dict(U_ev=U, W_ev=W, H=H, drives=DRIVES, python=platform.python_version(),
                  numpy=np.__version__, warning_options=sys.warnoptions,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='exact finite harmonic support; truncated H4 algebra only')
    status = 0
    try:
        self_check(report)
    except Exception as error:
        report['error'] = repr(error)
        status = 1
    print(json.dumps(dict(report, exit_code=status), indent=2, allow_nan=False))
    sys.exit(status)
