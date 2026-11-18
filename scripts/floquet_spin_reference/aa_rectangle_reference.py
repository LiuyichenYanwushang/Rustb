#!/usr/bin/env python3
"""AA bilayer spin and labelled BdG oracle, independent of Rustb.

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -W error aa_rectangle_reference.py
"""
from itertools import combinations
import json
from unittest.mock import patch

import numpy as np

import square_reference as sq


CELLS = np.array([[0, 0, 0], [0, 1, 0], [-1, 1, 0]])
Q_POINTS = np.array([[.17, .29, .37], [.31, .17, .37]])
LABELS = ('A1', 'B1', 'A2', 'B2')


def hp(matrix, sites, q):
    """Quadratic product-spinor Taylor coefficients in the x, ±y frames."""
    # The dense spin basis uses MSB at local site 0. Reference moments are
    # up/down/down/up in global order; local lowering flips either direction
    # with amplitude +1. Unused Fock sites are uncoupled spectators.
    masks = [1 << (3 - k) for k in range(len(sites))]
    v = sum(mask for mask, (i, _) in zip(masks, sites) if i in (1, 2))
    a, b = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
    for k, (i, ci) in enumerate(sites):
        f = v ^ masks[k]
        assert abs(matrix[f, v]) < 1e-15
        a[i, i] += matrix[f, f] - matrix[v, v]
        for l, (j, cj) in enumerate(sites):
            # Distinct sites may have the same orbital label, e.g. V ends.
            if k != l:
                phase = np.exp(2j * np.pi * np.dot(q, cj - ci))
                a[i, j] += phase * matrix[f, v ^ masks[l]]
                b[i, j] += phase * matrix[f ^ masks[l], v]
    return np.stack((a, b))


def run(sign, h, grid):
    nd, p, hops = sq.fock()
    pair, chi = sq.spin_operators()
    quartics = [pair[0, 1] @ pair[2, 3], pair[0, 3] @ pair[1, 2], pair[0, 2] @ pair[1, 3]]
    gram = [[np.vdot(a, b).real for b in quartics] for a in quartics]
    root3 = np.sqrt(3.0)
    rays = np.array([[0, -1, 0], [root3 / 2, .5, 0], [-root3 / 2, .5, 0]])
    z = np.array([0, 0, 1])
    zero = np.zeros(3, int)
    t, tp = .12, .09

    def blocks(displacements, weights):
        # Zero edge weights embed shorter open clusters into the four-site
        # Fock basis. Sample exp(-i a(θ)·d) directly, not Rust's coefficients.
        ds = np.zeros((4, 3))
        ds[:len(displacements)] = displacements
        ts = np.zeros(4)
        ts[:len(weights)] = weights

        def peierls(theta, _):
            cosine, sine = np.cos(theta), sign * np.sin(theta)
            light = np.stack((.4 * cosine, .3 * sine, .22 * cosine + .17 * sine), axis=1)
            return np.exp(-1j * (light @ ds.T))

        with patch.object(sq, 'peierls', peierls):
            h2, h4, _, _, gap = sq.sambe_orders(nd, p, hops * ts[:, None, None], h, None, grid)
        assert gap == 1
        return h2, h4

    # At <=4 hops a connected process uses a bond, a two-bond V, or a
    # four-cycle. Each mixed V is in exactly one rectangle. Each in-plane
    # bond belongs to one rectangle and four pure V's (two at each end), and
    # each vertical bond belongs to three rectangles. Count the full open
    # clusters, then subtract the excess dimers four and two times respectively.
    terms = []
    rectangles = []
    for d, r in zip(rays, CELLS):
        h2, h4 = blocks([d, z, -d, -z], [t, tp, t, tp])
        # Local cyclic order A1, B1, B2, A2 versus the output order A1, B1, A2, B2.
        terms.append((1, h2 + h4, [(0, zero), (1, r), (3, r), (2, zero)]))
        ring = np.linalg.solve(gram, [np.vdot(a, h4).real for a in quartics])
        np.testing.assert_allclose(ring, [ring[0], ring[0], -ring[0]], atol=1e-15, rtol=0)
        flip = h4[0b0101 ^ 0b1000, 0b0101 ^ 0b0010]
        assert abs(flip.imag) < 1e-15
        vertical_creation = h4[0b0101 ^ 0b1001, 0b0101]
        rectangles.append([sq.coefficient(pair[0, 1], h2), sq.coefficient(pair[1, 2], h2),
                           sq.coefficient(pair[0, 2], h4), *ring,
                           sq.coefficient(chi[0, 1, 2], h4), float(flip.real),
                           float(vertical_creation.imag)])

    for i, j in combinations(range(3), 2):
        wedge = sum(blocks([-rays[i], rays[j]], [t, t]))
        for a, b in ((0, 1), (2, 3)):
            terms.append((1, wedge, [(b, CELLS[i]), (a, zero), (b, CELLS[j])]))
            # B-centred wedge reverses both displacements, a half-period
            # shift of this single harmonic (zero net photon index).
            terms.append((1, wedge, [(a, -CELLS[i]), (b, zero), (a, -CELLS[j])]))
    for d, r in zip(rays, CELLS):
        dimer = sum(blocks([d], [t]))
        for a, b in ((0, 1), (2, 3)):
            terms.append((-4, dimer, [(a, zero), (b, r)]))
    vertical = sum(blocks([z], [tp]))
    for a, b in ((0, 2), (1, 3)):
        terms.append((-2, vertical, [(a, zero), (b, zero)]))
    assert len(terms) == 3 + 12 + 6 + 2

    matrices = []
    for q in Q_POINTS:
        a, b = sum(weight * hp(matrix, sites, q) for weight, matrix, sites in terms)
        am, bm = sum(weight * hp(matrix, sites, -q) for weight, matrix, sites in terms)
        np.testing.assert_allclose(bm, b.T, atol=1e-15, rtol=0)
        bdg = np.block([[a, b], [b.conj().T, am.conj()]])
        np.testing.assert_allclose(bdg, bdg.conj().T, atol=1e-15, rtol=0)
        matrices.append(bdg)
    return np.array(rectangles), np.array(matrices)


def main():
    coefficients, matrices, drift, reversal = {}, {}, {}, {}
    settings = [(4, 128), (6, 256)]
    for sign in (1, -1):
        coarse, wide = (run(sign, h, grid) for h, grid in settings)
        np.testing.assert_allclose(coarse[0], wide[0], atol=4e-15, rtol=0)
        np.testing.assert_allclose(coarse[1], wide[1], atol=2e-14, rtol=0)
        coefficients[sign] = {4: coarse[0].tolist(), 6: wide[0].tolist()}
        matrices[sign] = {4: coarse[1], 6: wide[1]}
        drift[sign] = dict(coefficients_ev=float(np.max(np.abs(coarse[0] - wide[0]))),
                           bdg_ev=np.max(np.abs(coarse[1] - wide[1]), axis=(1, 2)).tolist())

    # Reversing the sine terms conjugates the cluster spin matrices (equivalent
    # to θ -> π-θ plus conjugation). At fixed q this swaps Nambu sectors of
    # the Hessian; it is not same-q elementwise conjugation.
    perm = [4, 5, 6, 7, 0, 1, 2, 3]
    for h, _ in settings:
        forward, reverse = matrices[1][h], matrices[-1][h]
        swapped = forward[:, perm, :][:, :, perm]
        np.testing.assert_allclose(reverse, swapped, atol=1e-15, rtol=0)
        assert np.all(np.max(np.abs(reverse - forward), axis=(1, 2)) > 3e-7)
        reversal[h] = dict(sector_swap_residual_ev=np.max(np.abs(reverse - swapped), axis=(1, 2)).tolist(),
                           same_q_difference_ev=np.max(np.abs(reverse - forward), axis=(1, 2)).tolist())

    # Earlier signed rectangle-only anchors must agree with the complete sum.
    data = np.array(coefficients[1][4])
    phase = np.exp(2j * np.pi * (CELLS @ Q_POINTS[0]))
    a03 = phase @ data[:, -2]
    b02_imag = np.sum(data[:, -1])
    assert a03.imag > 5e-6 and b02_imag < -1e-7
    assert abs(matrices[1][4][0, 0, 3] - a03) < 1e-15
    assert abs(matrices[1][4][0, 0, 6].imag - b02_imag) < 1e-15
    print(json.dumps(dict(columns=['J_in_plane_H2', 'J_vertical_H2', 'J_cross_H4',
                                   'K1', 'K2', 'K3', 'chi012', 'flip02', 'creation03_imag'],
                          coefficients_ev=coefficients, A03_ev=[a03.real, a03.imag],
                          B02_imag_ev=float(b02_imag), q_fractional=Q_POINTS.tolist(),
                          nambu_labels=[f'{sector}:{s}' for sector in ('particle', 'hole') for s in LABELS],
                          cluster_counts=dict(rectangle=3, in_plane_v=12, in_plane_dimer=6,
                                              vertical_dimer=2, dimer_weights=[-4, -2]),
                          settings_H_M_grid=[[h, 2*h, grid] for h, grid in settings],
                          min_abs_Q_energy_ev=1., cutoff_drift=drift, reversal=reversal,
                          matrices_ev={sign: {h: np.stack((m.real, m.imag), axis=-1).tolist()
                                              for h, m in runs.items()}
                                       for sign, runs in matrices.items()}),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
