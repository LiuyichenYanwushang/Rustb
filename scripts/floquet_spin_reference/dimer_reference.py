#!/usr/bin/env python3
"""Independent, bounded half-filled Hubbard dimer reference; NumPy only."""
import argparse
import hashlib
import json
import os
import platform
import sys
import traceback

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fock():
    # spin-major modes (0 up, 1 up, 0 down, 1 down); ascending creators.
    states = [s for s in range(16) if bin(s).count('1') == 2]
    def cd_c(i, j):
        op = np.zeros((6, 6), complex)
        for col, s in enumerate(states):
            if not (s >> j & 1):
                continue
            sign = (-1) ** bin(s & ((1 << j) - 1)).count('1')
            r = s ^ (1 << j)
            if r >> i & 1:
                continue
            sign *= (-1) ** bin(r & ((1 << i) - 1)).count('1')
            op[states.index(r | (1 << i)), col] = sign
        return op
    forward = cd_c(0, 1) + cd_c(2, 3)  # bra site 0, ket site 1, d=+1
    doublons = np.array([int((s & 5) == 5) + int((s & 10) == 10)
                         for s in states])
    p = np.zeros((6, 4), complex)
    # Site tensor basis |up up>, |up down>, |down up>, |down down>.
    # c_0down^dag c_1up^dag is minus ascending spin-major Fock state.
    for col, (s, sign) in enumerate([(3, 1), (9, 1), (6, -1), (12, 1)]):
        p[states.index(s), col] = sign
    require(forward[states.index(5), states.index(6)] == 1 and
            forward[states.index(5), states.index(9)] == 1,
            'two-point bra/ket c-dagger-c direction/sign check')
    return states, forward, doublons, p


def coefficients(a, b, cutoff, samples=2048):
    theta = 2 * np.pi * np.arange(samples) / samples
    phase = a * np.cos(theta) + b * np.sin(2 * theta)
    # Direct quadrature, not a Bessel backend or Rust input API.
    return {m: np.mean(np.exp(1j * m * theta) * np.exp(-1j * phase))
            for m in range(-cutoff, cutoff + 1)}


def sambe(u, w, a, b, m):
    require(isinstance(m, int) and 1 <= m <= 10, 'photon budget: 1 <= M <= 10')
    states, forward, doublons, p6 = fock()
    cf = coefficients(a, b, 2 * m)
    cb = coefficients(-a, -b, 2 * m)
    photons = np.arange(-m, m + 1)
    h0 = (u * doublons[None, :] + w * photons[:, None]).ravel()
    v = np.zeros((len(h0), len(h0)), complex)
    for i, n in enumerate(photons):
        for j, k in enumerate(photons):
            v[6*i:6*i+6, 6*j:6*j+6] = (
                cf[int(n-k)] * forward + cb[int(n-k)] * forward.conj().T)
    p = np.zeros((len(h0), 4), complex)
    p[6*m:6*m+6] = p6
    q = np.real(np.diag(np.eye(len(h0)) - p @ p.conj().T)) > .5
    require(np.all(np.isfinite(h0[q])) and np.min(np.abs(h0[q])) > .5,
            'nonfinite/zero/near-resonant Q denominator')
    r = np.zeros_like(h0)
    # NumPy 1.20.3 on this host spuriously warns for vector division of
    # these 98 nonzero denominators (retained probe); scalar division avoids it.
    r[q] = [-1 / float(energy) for energy in h0[q]]
    require(np.all(np.isfinite(r)) and np.max(np.abs(h0[q]*r[q]+1)) < 3e-16,
            'invalid Q inverse or denominator inversion residual')
    return h0, v, p, r, cf, cb, states


def dense_orders(v, p, r):
    vp = v @ p
    h2 = vp.conj().T @ (r[:, None] * vp)
    norm = vp.conj().T @ ((r*r)[:, None] * vp)
    chain = vp.conj().T @ (r[:, None] * (v @ (r[:, None] * (
        v @ (r[:, None] * vp)))))
    folded = -.5 * (norm @ h2 + h2 @ norm)
    return h2, chain + folded, chain, folded


def tracked(h0, v, p, t):
    previous = p
    min_step = 1.
    min_p = 1.
    max_eigen_residual = 0.
    # ponytail: fixed 12-step continuation, not a general avoided-crossing solver.
    for amplitude in np.linspace(0, t, 13)[1:]:
        h = np.diag(h0) + amplitude * v
        energies, vectors = np.linalg.eigh(h)
        weights = np.sum(np.abs(previous.conj().T @ vectors)**2, axis=0)
        chosen = np.argsort(weights)[-4:]
        psi = vectors[:, chosen]
        step_sv = np.linalg.svd(previous.conj().T @ psi, compute_uv=False)
        min_step = min(min_step, float(step_sv[-1]))
        overlap = p.conj().T @ psi
        left, singular, right_h = np.linalg.svd(overlap)
        min_p = min(min_p, float(singular[-1]))
        require(step_sv[-1] > .95 and singular[-1] > .9,
                'lost isolated, full-rank P continuation')
        outside = np.ones(len(energies), bool)
        outside[chosen] = False
        separation = np.min(np.abs(energies[chosen, None] - energies[None, outside]))
        require(separation > .5, 'target subspace not spectrally separated')
        max_eigen_residual = max(max_eigen_residual, float(np.linalg.norm(
            h @ psi - psi * energies[chosen], ord=2)))
        previous = psi
    # O=P^dag psi=L s R^dag; U=psi R L^dag has P^dag U positive.
    rotation = right_h.conj().T @ left.conj().T
    canonical = rotation.conj().T @ np.diag(energies[chosen]) @ rotation
    iso = psi @ rotation
    boundary = float(np.linalg.norm(iso[:6], ord=2)**2 +
                     np.linalg.norm(iso[-6:], ord=2)**2)
    require(np.all(np.isfinite(canonical)) and max_eigen_residual < 1e-11,
            'nonfinite matrix or inaccurate eigensolve')
    require(np.linalg.norm(iso.conj().T @ iso-np.eye(4)) < 1e-12 and
            np.linalg.norm(p.conj().T @ iso-iso.conj().T @ p) < 1e-12,
            'canonical isometry/polar residual')
    return canonical, dict(min_step_overlap=min_step, min_p_singular=min_p,
                           boundary_weight=boundary, separation=float(separation),
                           eigen_residual=max_eigen_residual,
                           isometry_residual=float(np.linalg.norm(iso.conj().T @ iso-np.eye(4))),
                           polar_hermiticity=float(np.linalg.norm(p.conj().T @ iso-
                                                                iso.conj().T @ p)))


def spin_exchange():
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    return -np.outer(singlet, singlet)  # S0.S1 - 1/4, spin length 1/2


def run(report):
    require(os.environ.get('OPENBLAS_NUM_THREADS') == '1',
            'run with OPENBLAS_NUM_THREADS=1 before importing NumPy')
    with open(__file__, 'rb') as source:
        source_hash = hashlib.sha256(source.read()).hexdigest()
    report['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
        openblas_threads=os.environ['OPENBLAS_NUM_THREADS'], command=sys.argv,
        source_sha256=source_hash)
    report['parameters'] = dict(U=4., W=3., A=.8, B=.45, M=[8, 10],
                                t=[.16, .08, .04], samples=[2048, 4096], steps=12,
                                maximum_dimension=126)
    u, w = 4., 3.
    exchange = spin_exchange()
    h0, v, p, r, _, _, states = sambe(u, w, 0., 0., 2)
    h2, h4, chain, folded = dense_orders(v, p, r)
    report['static'] = []
    require(np.linalg.norm(h2 - 4/u*exchange) < 1e-13, 'static H2/sign')
    require(np.linalg.norm(h4 + 16/u**3*exchange) < 1e-13, 'static H4/folded')
    require(np.linalg.norm(chain) < 1e-13 and np.linalg.norm(folded) > .1,
            'static fourth order must be folded, not a bare chain')
    for t in [.16, .08, .04]:
        exact, diagnostic = tracked(h0, v, p, t)
        # Rationalized closed form avoids subtractive cancellation.
        gap = 8*t*t / (np.sqrt(u*u + 16*t*t) + u)
        residual = float(np.linalg.norm(exact-gap*exchange, ord=2))
        report['static'].append(dict(t=t, gap=gap, exact_residual=residual, **diagnostic))
        require(residual < 2e-13, 'independent analytic static gap mismatch')
    h0, v, p, r, cf, cb, states = sambe(u, w, .8, .45, 10)
    h2, h4, chain, folded = dense_orders(v, p, r)
    hermitian = float(np.linalg.norm(v-v.conj().T))
    require(hermitian < 1e-13, 'reverse link/Sambe hermiticity')
    c2 = coefficients(.8, .45, 20, 4096)
    dft_residual = max(abs(cf[n]-c2[n]) for n in cf)
    require(dft_residual < 1e-14, 'DFT grid convergence')
    k2 = 2*sum((abs(cf[n])**2+abs(cb[n])**2)/(u+n*w) for n in range(-10, 11))
    wrong = 4*sum(abs(cf[n])**2/(u+n*w) for n in range(-10, 11))
    reverse = 4*sum(abs(cb[n])**2/(u+n*w) for n in range(-10, 11))
    q = np.any(np.abs(p) > 0, axis=1) == False
    report['denominators'] = dict(min_abs_Q=float(np.min(np.abs(h0[q]))),
        inverse_residual=float(np.max(np.abs(h0[q]*r[q]+1))),
        finite=bool(np.all(np.isfinite(r))))
    report['two_direction'] = dict(K2=k2, wrong_forward=wrong, wrong_reverse=reverse,
        asymmetry=max(abs(abs(cf[n])-abs(cf[-n])) for n in cf),
        dense_residual=float(np.linalg.norm(h2-k2*exchange)),
        reverse_average_residual=float(abs(k2-(wrong+reverse)/2)),
        dft_residual=float(dft_residual), sambe_hermiticity=hermitian)
    require(abs(k2-wrong) > .01 and abs(k2-(wrong+reverse)/2) < 1e-13,
            'two-frequency counterexample or orientation invariance failed')
    require(np.linalg.norm(h2-k2*exchange) < 1e-13, 'two-direction K2 mismatch')
    # Explicitly remove Nd=0, n!=0 from R as a negative control, not our oracle.
    r_doublon_only = r.copy()
    for index in range(len(r)):
        s = states[index % 6]
        if (s & 5) != 5 and (s & 10) != 10:
            r_doublon_only[index] = 0
    _, incomplete, _, _ = dense_orders(v, p, r_doublon_only)
    report['fourth_order'] = dict(chain_norm=float(np.linalg.norm(chain)),
        folded_norm=float(np.linalg.norm(folded)),
        nonzero_photon_single_contribution=float(np.linalg.norm(h4-incomplete)),
        hermiticity=float(np.linalg.norm(h4-h4.conj().T)))
    require(np.linalg.norm(h4-incomplete) > 1e-5, 'Q negative control ineffective')
    require(np.linalg.norm(h4-h4.conj().T) < 1e-13, 'H4 Hermiticity')
    report['finite_drive'] = []
    for t in [.16, .08, .04]:
        exact, diagnostic = tracked(h0, v, p, t)
        row = dict(t=t, residual_H2=float(np.linalg.norm(exact-t*t*h2, ord=2)),
                   residual_H24=float(np.linalg.norm(exact-t*t*h2-t**4*h4, ord=2)),
                   **diagnostic)
        report['finite_drive'].append(row)
    rows = report['finite_drive']
    for first, second in zip(rows, rows[1:]):
        ratio2 = first['residual_H2']/second['residual_H2']
        ratio4 = first['residual_H24']/second['residual_H24']
        second['halving_ratio_H2'] = ratio2
        second['halving_ratio_H24'] = ratio4
        require(14 < ratio2 < 18 and 50 < ratio4 < 80, 'finite-drive t^4/t^6 scaling')
    small = sambe(u, w, .8, .45, 8)
    exact8, diag8 = tracked(*small[:3], .16)
    exact10, _ = tracked(h0, v, p, .16)
    h28, h48, _, _ = dense_orders(small[1], small[2], small[3])
    report['convergence'] = dict(matrix_M8_M10=float(np.linalg.norm(exact8-exact10, ord=2)),
        H2_M8_M10=float(np.linalg.norm(h28-h2)), H4_M8_M10=float(np.linalg.norm(h48-h4)),
        boundary_M8=diag8['boundary_weight'])
    require(report['convergence']['matrix_M8_M10'] < 1e-9 and
            report['convergence']['H2_M8_M10'] < 1e-8 and
            report['convergence']['H4_M8_M10'] < 1e-7, 'M convergence')
    require(diag8['boundary_weight'] < 1e-8, 'boundary weight convergence')
    # Lowest raw eigenvalues sit at negative photon copies, not our zero-copy P.
    _, vectors = np.linalg.eigh(np.diag(h0)+.16*v)
    lowest_weight = float(np.linalg.norm(p.conj().T @ vectors[:, :4])**2)
    report['wrong_lowest_four_P_weight'] = lowest_weight
    require(lowest_weight < .01, 'lowest-spectrum negative control ineffective')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    report = {}
    try:
        run(report)
        report['status'] = 'PASS: T20a reference checks only; T20/M1 not certified'
        code = 0
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        code = 1
    report['exit_code'] = code
    print(json.dumps(report, indent=2, allow_nan=False))
    sys.exit(code)
