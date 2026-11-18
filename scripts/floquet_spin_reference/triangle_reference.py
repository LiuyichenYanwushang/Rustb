#!/usr/bin/env python3
"""Independent 3-site Hubbard triangle: 20-state Fock and P8 spin reference."""
from __future__ import print_function
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import traceback

import numpy as np

NMODE = 6
DIM = 20
U_DEFAULT = 4.0
PHI_DEFAULT = 0.41
T_VALUES = (0.1, 0.05, 0.025)


def popcount(value):
    return bin(value).count("1")


def fock_basis():
    return [sum(1 << i for i in occupied) for occupied in
            __import__("itertools").combinations(range(NMODE), 3)]


def apply_c(state, mode, create):
    occupied = (state >> mode) & 1
    if (create and occupied) or (not create and not occupied):
        return None
    sign = -1 if popcount(state & ((1 << mode) - 1)) % 2 else 1
    return state ^ (1 << mode), sign


def bilinear(state, create_mode, annihilate_mode):
    first = apply_c(state, annihilate_mode, False)
    if first is None:
        return None
    second = apply_c(first[0], create_mode, True)
    if second is None:
        return None
    return second[0], first[1] * second[1]


def site_of(mode):
    return mode % 3


def make_fock():
    basis = fock_basis()
    lookup = dict((state, i) for i, state in enumerate(basis))
    h0 = np.zeros((DIM, DIM), dtype=np.complex128)
    hop = np.zeros_like(h0)
    for col, state in enumerate(basis):
        doubles = sum(1 for site in range(3)
                      if ((state >> site) & 1) and ((state >> (site + 3)) & 1))
        h0[col, col] = U_DEFAULT * doubles
        for spin in range(2):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                # t_ij=<i|H|j>; directed cycle product is t01*t12*t20.
                for dst, src, amplitude in ((i, j, 1.0), (j, i, 1.0)):
                    result = bilinear(state, dst + 3 * spin, src + 3 * spin)
                    if result is not None:
                        row = lookup[result[0]]
                        hop[row, col] += result[1] * amplitude
    return basis, lookup, h0, hop


def p_embedding(basis, lookup):
    # Site tensor order |s0,s1,s2>; spin-major Fock creators are up modes
    # first, then down. Compute the sign rather than assume a convention.
    p = np.zeros((DIM, 8), dtype=np.complex128)
    for bits in range(8):
        modes = [site + (3 if ((bits >> (2 - site)) & 1) else 0) for site in range(3)]
        state = sum(1 << mode for mode in modes)
        inversions = sum(1 for i in range(3) for j in range(i + 1, 3)
                         if modes[i] > modes[j])
        sign = -1 if inversions % 2 else 1
        p[lookup[state], bits] = sign
    if not np.allclose(np.dot(p.conj().T, p), np.eye(8), atol=1e-14, rtol=0):
        raise RuntimeError("P embedding is not orthonormal")
    return p


def hopping(phi):
    # The index product t_01*t_12*t_20 is exp(i phi). Since t_ij multiplies
    # c_i^dag c_j, electron arrows run oppositely: 1->0, 2->1, 0->2.
    return {(0, 1): np.exp(1j * phi / 3.0),
            (1, 2): np.exp(1j * phi / 3.0),
            (2, 0): np.exp(1j * phi / 3.0)}


def make_flux_hop(basis, lookup, phi):
    matrix = np.zeros((DIM, DIM), dtype=np.complex128)
    links = hopping(phi)
    for col, state in enumerate(basis):
        for spin in range(2):
            for (i, j), value in links.items():
                for dst, src, amplitude in ((i, j, value), (j, i, value.conjugate())):
                    result = bilinear(state, dst + 3 * spin, src + 3 * spin)
                    if result is not None:
                        matrix[lookup[result[0]], col] += result[1] * amplitude
    if not np.allclose(matrix, matrix.conj().T, atol=1e-14, rtol=0):
        raise RuntimeError("hopping is not Hermitian")
    return matrix


def effective_coefficients(h0, v, p):
    q = np.eye(DIM, dtype=np.complex128) - np.dot(p, p.conj().T)
    # H0 is diagonal and invertible on Q. Scalar division avoids NumPy 1.20
    # vector divide warnings under strict -Werror.
    r = np.zeros_like(h0)
    for i in range(DIM):
        if abs(q[i, i]) > 0.5:
            denom = -float(h0[i, i].real)
            if not math.isfinite(denom) or abs(denom) < 1e-12:
                raise RuntimeError("non-finite or zero Q denominator")
            r[i, i] = complex(1.0 / denom, 0.0)
    if not np.all(np.isfinite(r)):
        raise RuntimeError("resolvent contains non-finite entries")
    h2 = np.linalg.multi_dot((p.conj().T, v, r, v, p))
    h3 = np.linalg.multi_dot((p.conj().T, v, r, v, r, v, p))
    h2, anti2 = hermitize(h2, "H2")
    h3, anti3 = hermitize(h3, "H3")
    return h2, h3, r, anti2, anti3


def hermitize(matrix, label):
    antihermitian_residual = float(np.linalg.norm(matrix - matrix.conj().T))
    scale = max(float(np.linalg.norm(matrix)), 1e-14)
    if not math.isfinite(antihermitian_residual) or antihermitian_residual > 5e-11 * scale:
        raise RuntimeError("%s anti-Hermitian residual %.6g exceeds tolerance %.6g" %
                           (label, antihermitian_residual, 5e-11 * scale))
    return 0.5 * (matrix + matrix.conj().T), antihermitian_residual


def spin_operators():
    identity = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) * 0.5
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) * 0.5
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) * 0.5
    return (identity, sx, sy, sz)


def tensor_operators():
    local = spin_operators()
    result = {}
    for a in range(4):
        for b in range(4):
            for c in range(4):
                result[(a, b, c)] = np.kron(np.kron(local[a], local[b]), local[c])
    return result


def chirality_operator(tensors):
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
           (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    chi = np.zeros((8, 8), dtype=np.complex128)
    for (a, b, c), sign in eps.items():
        chi += sign * tensors[(a + 1, b + 1, c + 1)]
    return chi


def decompose(matrix, tensors):
    coefficients = {}
    for key, operator in tensors.items():
        gram = np.trace(operator.conj().T.dot(operator)) / 8.0
        if abs(gram) < 1e-14:
            raise RuntimeError("tensor basis has a singular Gram element")
        coefficients[key] = (np.trace(operator.conj().T.dot(matrix)) / 8.0) / gram
    reconstruction = sum((value * tensors[key] for key, value in coefficients.items()),
                         np.zeros((8, 8), dtype=np.complex128))
    return coefficients, float(np.linalg.norm(matrix - reconstruction))


def polar_frame(vectors, p):
    overlap = np.dot(p.conj().T, vectors)
    left, singular, right_h = np.linalg.svd(overlap, full_matrices=False)
    if singular.shape[0] != 8 or singular[-1] < 0.80:
        raise RuntimeError("tracked P8 lost isolation: min singular=%g" % singular[-1])
    # If P^dagger Psi = L Sigma R^dagger, the canonical frame is
    # Psi polar(P^dagger Psi)^dagger = Psi R L^dagger.
    frame = np.dot(vectors, np.dot(right_h.conj().T, left.conj().T))
    canonical_overlap = np.dot(p.conj().T, frame)
    polar_residual = float(np.linalg.norm(canonical_overlap - canonical_overlap.conj().T))
    polar_eigen_min = float(np.min(np.linalg.eigvalsh(0.5 *
                                  (canonical_overlap + canonical_overlap.conj().T))))
    isometry_residual = float(np.linalg.norm(np.dot(frame.conj().T, frame) - np.eye(8)))
    if polar_residual > 1e-11 or polar_eigen_min < 0.0 or isometry_residual > 1e-11:
        raise RuntimeError("canonical polar frame failed residual checks")
    return frame, float(singular[-1]), polar_residual, polar_eigen_min, isometry_residual


def exact_low_block(h0, v, p, t):
    previous = p.copy()
    min_overlap = 1.0
    min_p_overlap = 1.0
    min_gap = float("inf")
    boundary = 0.0
    for scale in np.linspace(0.0, t, 11)[1:]:
        energies, vectors = np.linalg.eigh(h0 + scale * v)
        # This is a rolling continuation: select the eight eigenvectors with
        # greatest total principal overlap with the preceding tracked frame.
        continuation_weights = np.sum(np.abs(np.dot(previous.conj().T, vectors)) ** 2,
                                      axis=0)
        chosen = np.argsort(continuation_weights)[-8:]
        selected = vectors[:, chosen]
        cross = np.linalg.svd(np.dot(previous.conj().T, selected), compute_uv=False)
        minimum = float(cross[-1])
        min_overlap = min(min_overlap, minimum)
        if minimum < 0.80:
            raise RuntimeError("continuation overlap below .80: %g" % minimum)
        # Separately measure retained overlap with the original P sector.
        p_weights = np.sum(np.abs(np.dot(p.conj().T, vectors)) ** 2, axis=0)
        p_singular = np.linalg.svd(np.dot(p.conj().T, selected), compute_uv=False)
        min_p_overlap = min(min_p_overlap, float(p_singular[-1]))
        if p_singular[-1] < 0.80:
            raise RuntimeError("tracked subspace P overlap below .80: %g" % p_singular[-1])
        boundary = max(boundary, max(0.0,
                                     float(np.max(1.0 - p_weights[np.array(chosen)]))))
        # Check spectral isolation from the other 12 eigenvalues.
        chosen_set = set(int(i) for i in chosen)
        outside = [i for i in range(DIM) if i not in chosen_set]
        gap = float(np.min(np.abs(energies[np.array(outside)][:, None] -
                                 energies[np.array(chosen)][None, :])))
        min_gap = min(min_gap, gap)
        if gap < 0.25:
            raise RuntimeError("P8/complement gap below .25 eV: %g" % gap)
        previous = selected
    frame, singular, polar_residual, polar_eigen_min, isometry_residual = polar_frame(previous, p)
    if not np.all(np.isfinite(frame)):
        raise RuntimeError("polar frame is non-finite")
    h_final = h0 + t * v
    raw_block = np.linalg.multi_dot((frame.conj().T, h_final, frame))
    block, block_anti_residual = hermitize(raw_block, "ED projected block")
    eigen_residual = float(np.linalg.norm(np.dot(h_final, frame) - np.dot(frame, raw_block)))
    if eigen_residual > 1e-10:
        raise RuntimeError("ED tracked-subspace eigen residual too large: %g" % eigen_residual)
    return block, {"min_principal_overlap": min_overlap,
                   "min_p_overlap_singular": min_p_overlap,
                   "final_p_overlap_singular": singular,
                   "min_isolation_gap_ev": min_gap,
                   "max_selected_nonP_weight": boundary,
                   "polar_overlap_hermiticity_residual": polar_residual,
                   "polar_overlap_min_eigenvalue": polar_eigen_min,
                   "isometry_residual": isometry_residual,
                   "ed_projected_block_antihermiticity_residual": block_anti_residual,
                   "tracked_subspace_eigen_residual": eigen_residual}


def off_scalar_phi_coefficients(coeffs):
    return dict((key, value) for key, value in coeffs.items()
                if key != (0, 0, 0) and sum(1 for x in key if x) >= 2)


def run():
    basis, lookup, h0, _ = make_fock()
    p = p_embedding(basis, lookup)
    t_ops = tensor_operators()
    chi = chirality_operator(t_ops)
    if abs(np.trace(chi.conj().T.dot(chi)).real / 8.0 - 3.0 / 32.0) > 1e-14:
        raise RuntimeError("spin-1/2 chirality normalization mismatch")
    records = []
    phi = PHI_DEFAULT
    for flux in (0.0, phi, -phi):
        v = make_flux_hop(basis, lookup, flux)
        h2, h3, r, anti2, anti3 = effective_coefficients(h0, v, p)
        c2, err2 = decompose(h2, t_ops)
        c3, err3 = decompose(h3, t_ops)
        cchi = np.trace(chi.conj().T.dot(h3)) / np.trace(chi.conj().T.dot(chi))
        if abs(cchi.imag) > 2e-11:
            raise RuntimeError("Hermitian chirality coefficient is not real")
        records.append({"phi": flux, "h2": h2, "h3": h3, "c2": c2,
                        "c3": c3, "decomp_h2": err2, "decomp_h3": err3,
                        "anti_h2": anti2, "anti_h3": anti3,
                        "chi_h3": float(cchi.real), "r": r})
    zero, plus, minus = records
    # At zero flux, particle-hole symmetry forces the entire third-order block
    # to vanish; check the computed matrix, not only its chirality projection.
    if abs(zero["chi_h3"]) > 2e-11:
        raise RuntimeError("zero-flux third-order chirality did not cancel")
    if abs(float(np.trace(zero["h3"]).real / 8.0)) > 2e-11:
        raise RuntimeError("unexpected zero-flux H3 scalar; inspect PH cancellation")
    if np.linalg.norm(zero["h3"]) > 2e-11:
        raise RuntimeError("zero-flux H3 has a nonzero tensor component; inspect PH cancellation")
    if abs(plus["chi_h3"] + minus["chi_h3"]) > 2e-11:
        raise RuntimeError("chirality coefficient is not odd under flux reversal")
    # For the declared t_01 t_12 t_20 orientation and spin-major c†_i c_j
    # convention the three-site coefficient is -24 sin(Phi)/U^2.
    expected_chi = -24.0 * math.sin(phi) / (U_DEFAULT * U_DEFAULT)
    if abs(plus["chi_h3"] - expected_chi) > 2e-11:
        raise RuntimeError("directed 24-factor chirality audit failed: got %.16g expected %.16g" %
                           (plus["chi_h3"], expected_chi))
    product = np.prod([hopping(phi)[edge] for edge in ((0, 1), (1, 2), (2, 0))])
    if abs(product - np.exp(1j * phi)) > 1e-14:
        raise RuntimeError("directed flux product convention failed")
    # Exact ED convergence at one fixed nonzero flux; matrix norm expansion
    # orders are measured without fitting coefficients.
    scaling = []
    h2, h3, _, _, _ = effective_coefficients(h0, make_flux_hop(basis, lookup, phi), p)
    for t in T_VALUES:
        exact, diag = exact_low_block(h0, make_flux_hop(basis, lookup, phi), p, t)
        residual2 = float(np.linalg.norm(exact - t * t * h2))
        residual23 = float(np.linalg.norm(exact - t * t * h2 - t * t * t * h3))
        scaling.append({"t": t, "residual_h2": residual2,
                        "residual_h23": residual23, "diagnostics": diag})
    ratios = []
    for key in ("residual_h2", "residual_h23"):
        ratios.append([scaling[i][key] / scaling[i + 1][key] for i in range(2)])
    if not all(math.isfinite(x) and 7.0 < x < 9.0 for x in ratios[0]):
        raise RuntimeError("H2 residual ratio is outside cubic window (7,9): %r" % ratios[0])
    if not all(math.isfinite(x) and 14.0 < x < 18.0 for x in ratios[1]):
        raise RuntimeError("H2+H3 residual ratio is outside quartic window (14,18): %r" % ratios[1])
    return {"basis_states": DIM, "p_dimension": 8, "spin_order": "spin-major (up sites 0..2, down sites 0..2)",
            "fock_embedding_signs": [int(np.real(p[i, j])) for j in range(8)
                                     for i in range(DIM) if abs(p[i, j]) > 0.5],
            "parameters": {"U": U_DEFAULT, "phi": phi, "t_values": list(T_VALUES)},
            "directed_hopping_product": [float(product.real), float(product.imag)],
            "flux_results": [{"phi": row["phi"], "chi_h3": row["chi_h3"],
                              "trace_h3_over8": float(np.trace(row["h3"]).real / 8.0),
                              "h2_antihermitian_residual": row["anti_h2"],
                              "h3_antihermitian_residual": row["anti_h3"],
                              "h2_decomposition_error": row["decomp_h2"],
                              "h3_decomposition_error": row["decomp_h3"],
                              "h2_tensor_coefficients": {"%s%s%s" % key: [float(value.real), float(value.imag)]
                                                          for key, value in row["c2"].items()
                                                          if abs(value) > 1e-12},
                              "h3_scalar_constant": [float(row["c3"][(0, 0, 0)].real),
                                                     float(row["c3"][(0, 0, 0)].imag)],
                              "h3_coefficients": {"%s%s%s" % key: [float(value.real), float(value.imag)]
                                                  for key, value in off_scalar_phi_coefficients(row["c3"]).items()
                                                  if abs(value) > 1e-12}}
                             for row in records],
            "exact_scaling": scaling, "halving_ratios": ratios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.parse_args()
    try:
        result = run()
        with open(__file__, "rb") as source_file:
            source_hash = hashlib.sha256(source_file.read()).hexdigest()
        result.update({"python": platform.python_version(), "numpy": np.__version__,
                       "source_sha256": source_hash, "exit_code": 0})
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception:
        with open(__file__, "rb") as source_file:
            source_hash = hashlib.sha256(source_file.read()).hexdigest()
        print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                          "source_sha256": source_hash,
                          "exit_code": 1, "traceback": traceback.format_exc()}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
