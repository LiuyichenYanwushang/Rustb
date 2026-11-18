#!/usr/bin/env python3
"""Dense Fock/Sambe oracle for the noncoplanar four-site loop and its bridges.

Run with OPENBLAS_NUM_THREADS=1 python3 -B -W error nonplanar_reference.py.
The reported signed spin operators and spectrum do not call Rustb.
"""
import json
from unittest.mock import patch

import numpy as np

import square_reference as sq


def hp(matrix, sites, q):
    """Product-spinor Taylor coefficients from direct flip matrix elements."""
    flip = [1 << (3 - i) for i in range(len(sites))]
    v = sum(f for f, (i, _) in zip(flip, sites) if i % 2)
    a, b = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
    for k, (i, ci) in enumerate(sites):
        f = v ^ flip[k]
        assert abs(matrix[f, v]) < 1e-15
        a[i, i] += matrix[f, f] - matrix[v, v]
        for l, (j, cj) in enumerate(sites):
            if k != l:
                phase = np.exp(2j * np.pi * np.dot(q, cj - ci))
                a[i, j] += phase * matrix[f, v ^ flip[l]]
                b[i, j] += phase * matrix[f ^ flip[l], v]
    return np.stack((a, b))


def run(z, h, grid):
    nd, p, hops = sq.fock()
    pair, chi = sq.spin_operators()
    quartic = [pair[0, 1] @ pair[2, 3], pair[0, 2] @ pair[1, 3], pair[0, 3] @ pair[1, 2]]
    gram = [[np.vdot(a, b).real for b in quartic] for a in quartic]
    lat = 4 * np.eye(3)
    orb = np.array([[.25, 0, .875], [.25, .25, .875], [.5, .25, .125], [0, 0, .125]])
    c = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]])
    y = np.array([0, 1, 0])
    xyz = (orb + c) @ lat
    assert np.linalg.det(xyz[1:] - xyz[0]) == -2
    ring = list(zip(np.roll(xyz, -1, axis=0) - xyz, (1., .8, .6, .9)))
    bridge = (xyz[0] + y @ lat - xyz[1], .5)
    assert np.array_equal(bridge[0], [0, 3, 0])
    zero = (np.zeros(3), 0.)
    t = .1
    light = np.array([.4, 0., 1j * z])

    def blocks(edges):
        ds, weights = map(np.array, zip(*(edges + [zero] * (4 - len(edges)))))

        def peierls(theta, _):
            a = np.real(np.exp(-1j * theta[:, None]) * light[None, :])
            return np.exp(-1j * (a @ ds.T))

        with patch.object(sq, 'peierls', peierls):
            h2, h4, _, _, gap = sq.sambe_orders(nd, p, hops * weights[:, None, None], h, None, grid)
        assert gap == 1
        return h2, h4

    def full(edges):
        a, b = blocks(edges)
        return t**2 * a + t**4 * b

    a, b = blocks(ring)
    clusters = [(t**2 * a + t**4 * b, list(enumerate(c))),
                (full([bridge]), [(1, c[1]), (0, y+c[0])])]

    def reverse(edge):
        return -edge[0], edge[1]

    # A cut edge occurs only in its dimer or in two-bond wedges through four
    # hops. Remove both embedded dimers from each of its four endpoint wedges.
    for e, f, sites in [(ring[0], bridge, [(0, c[0]), (1, c[1]), (0, y+c[0])]),
                        (reverse(ring[1]), bridge, [(2, c[2]), (1, c[1]), (0, y+c[0])]),
                        (bridge, ring[0], [(1, c[1]), (0, y+c[0]), (1, y+c[1])]),
                        (bridge, reverse(ring[3]), [(1, c[1]), (0, y+c[0]), (3, y+c[3])])]:
        wedge = full([e, f]) - full([e]) - full([zero, f])
        assert np.linalg.norm(wedge.imag) < 1e-15
        clusters.append((wedge, sites))
    chiralities = [sq.coefficient(op, b) for op in chi.values()]
    quartics = np.linalg.solve(gram, [np.vdot(op, b).real for op in quartic])
    rest = b - sum(v * op for v, op in zip(chiralities, chi.values()))
    rest -= sum(v * op for v, op in zip(quartics, quartic))
    constant = float(np.trace(rest).real / 16)
    pairs = [sq.coefficient(op, rest) for op in pair.values()]
    assert np.linalg.norm(rest - constant * np.eye(16) - sum(v * op for v, op in zip(pairs, pair.values()))) < 1e-12
    out = dict(H=h, M=2*h, DFT_grid=grid, H4=dict(chi=chiralities, quartic=quartics.tolist(), pair=pairs,
               constant=constant), magnon=[])
    for q in (np.array([.19, .23, .37]), np.array([.19, .41, .37])):
        aq, bq = sum(hp(m, s, q) for m, s in clusters)
        aq_minus, _ = sum(hp(m, s, -q) for m, s in clusters)
        bdg = np.block([[aq, bq], [bq.conj().T, aq_minus.conj()]])
        assert np.linalg.norm(bdg - bdg.conj().T) < 1e-15
        values, vectors = np.linalg.eigh(bdg)
        assert values[0] > 1e-4
        root = (vectors * np.sqrt(values)) @ vectors.conj().T
        bands = np.linalg.eigvalsh(root @ np.diag([1.]*4 + [-1.]*4) @ root)[4:]
        out['magnon'].append(dict(q=q.tolist(), min_bdg_eigenvalue_ev=float(values[0]), bands_ev=bands.tolist(),
                    matrix_ev=np.stack((bdg.real, bdg.imag), axis=-1).tolist()))
    return out, a, b


def main():
    cases = {}
    for z in [.35, 0., -.35]:
        coarse, ac, bc = run(z, 4, 128)
        fine, af, bf = run(z, 6, 256)
        fine['cutoff_frobenius_2_4'] = [float(np.linalg.norm(ac - af)), float(np.linalg.norm(bc - bf))]
        h2, h4 = fine['cutoff_frobenius_2_4']
        # The dense coefficients do not include t^2 and t^4, respectively.
        assert .1**2 * h2 < 5e-12 and .1**4 * h4 < 1e-12
        cases[str(z)] = {'4': coarse, '6': fine}
    pos, zero, neg = (cases[z]['4']['H4'] for z in ['0.35', '0.0', '-0.35'])
    assert pos['chi'][0] < -.15
    np.testing.assert_allclose(pos['chi'], -np.array(neg['chi']), atol=1e-13, rtol=0)
    np.testing.assert_allclose(zero['chi'], 0.0, atol=1e-13, rtol=0)
    assert abs(pos['quartic'][0] - zero['quartic'][0]) > .1
    print(json.dumps(cases, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
