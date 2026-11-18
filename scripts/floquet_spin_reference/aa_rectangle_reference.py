#!/usr/bin/env python3
"""Driven vertical AA rectangles: independent signed Fock/Sambe reference."""
import json
from unittest.mock import patch

import numpy as np

import square_reference as sq


def main():
    # Cyclic order A1, B1, B2, A2. In-plane bond length and layer spacing 1.
    nd, p, hops = sq.fock()
    hops = hops * np.array([.12, .09, .12, .09])[:, None, None]
    pair, chi = sq.spin_operators()
    quartics = [pair[0, 1] @ pair[2, 3], pair[0, 3] @ pair[1, 2], pair[0, 2] @ pair[1, 3]]
    gram = [[np.vdot(a, b).real for b in quartics] for a in quartics]
    root3 = np.sqrt(3.0)
    rays = [(0, -1), (root3 / 2, .5), (-root3 / 2, .5)]
    cells = np.array([[0, 0], [0, 1], [-1, 1]])
    output = {}
    for h, grid in [(4, 128), (6, 256)]:
        rectangles = []
        for dx, dy in rays:
            def peierls(theta, _):
                sine, cosine = np.sin(theta), np.cos(theta)
                xy = .4 * dx * cosine + .3 * dy * sine
                z = .22 * cosine + .17 * sine
                return np.exp(-1j * np.stack((xy, z, -xy, -z), axis=1))

            with patch.object(sq, 'peierls', peierls):
                h2, h4, _, _, gap = sq.sambe_orders(nd, p, hops, h, None, grid)
            assert gap == 1
            ring = np.linalg.solve(gram, [np.vdot(a, h4).real for a in quartics])
            np.testing.assert_allclose(ring, [ring[0], ring[0], -ring[0]], atol=1e-15, rtol=0)
            flip = h4[0b0101 ^ 0b1000, 0b0101 ^ 0b0010]
            assert abs(flip.imag) < 1e-15
            # Simultaneous creation on A1/A2 (bit masks 8 and 1).
            # Its imaginary part excludes all real exchange and ring terms.
            vertical_creation = h4[0b0101 ^ 0b1001, 0b0101]
            rectangles.append([sq.coefficient(pair[0, 1], h2), sq.coefficient(pair[1, 2], h2),
                               sq.coefficient(pair[0, 2], h4), *ring,
                               sq.coefficient(chi[0, 1, 2], h4), float(flip.real),
                               float(vertical_creation.imag)])
        output[h] = rectangles
    np.testing.assert_allclose(output[4], output[6], atol=4e-15, rtol=0)
    phase = np.exp(2j * np.pi * (cells @ np.array([.17, .29])))
    a03 = phase @ np.array(output[4])[:, -2]
    b02_imag = np.sum(np.array(output[4])[:, -1])
    assert a03.imag > 5e-6 and b02_imag < -1e-7
    print(json.dumps(dict(columns=['J_in_plane_H2', 'J_vertical_H2', 'J_cross_H4',
                                   'K1', 'K2', 'K3', 'chi012', 'flip02', 'creation03_imag'],
                          coefficients_ev=output, A03_ev=[a03.real, a03.imag],
                          B02_imag_ev=float(b02_imag)),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
