"""Generate the TB2J non-collinear reference used to calibrate the spinor
exchange extraction.

Builds a two-site spinor model (one orbital per site, spin-major basis
`[s0_up, s0_dn, s1_up, s1_dn]`) with a nearest-neighbour hopping `t` and a local
exchange field of magnitude `m` whose direction differs between the sites
(`z` on site 0, `x` on site 1), writes the Wannier90 files TB2J needs, and runs
TB2J's spinor path:

    python3 gen_reference.py
    wann2J.py --path . --spinor --prefix_spinor wannier90 --groupby spin \
        --elements Fe --efermi 0.0 --nz 60 --kmesh 6 6 1

`wannier90_centres.xyz` must use the symbol `X` for every Wannier centre,
because HamiltonIO's parser keeps only `X` entries. The `.win` needs the
`BEGIN UNIT_CELL_CART` / `BEGIN ATOMS_FRAC` blocks; a plain `lattice =` line is
rejected by TB2J's parser.

Reference output (TB2J 0.9.12.26, in meV, for the `Fe1-Fe2` bond at `R = 0`):

    J_iso = 0.9450
    DMI   = (0.0000, 1.8671, 0.0000)
    J_ani = [[1.182, 0.000, 1.182],
             [0.000, -5.739, 0.000],
             [1.182, 0.000, 1.182]]

and for `R = (-1, 0, 0)`: `J_iso = -0.0035`, `DMI = (0, -0.0027, 0)`.
TB2J prints `J_ani` and `DMI` in each site's local frame, so a Rust
implementation must apply the same frame rotation before comparing.
"""

import numpy as np

POSITIONS = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 0.0, 0.5]])
LATTICE = np.diag([3.0, 3.0, 20.0])
HOPPING = 0.1
FIELD = 1.0
DIRECTIONS = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
PAULI = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]]),
]


def on_site(site):
    direction = DIRECTIONS[site]
    return 0.5 * FIELD * sum(direction[i] * PAULI[i] for i in range(3))


def main():
    zeros = np.zeros((2, 2), dtype=complex)
    identity = np.eye(2)
    onsite = np.block(
        [[on_site(0), HOPPING * identity], [HOPPING * identity, on_site(1)]]
    )
    forward = np.block([[zeros, HOPPING * identity], [zeros, zeros]])
    backward = np.block([[zeros, zeros], [HOPPING * identity, zeros]])
    hoppings = [((0, 0, 0), onsite), ((1, 0, 0), forward), ((-1, 0, 0), backward)]
    n_orb = 4
    with open("wannier90_hr.dat", "w") as handle:
        handle.write("tb2j non-collinear reference\n")
        handle.write(f"  {n_orb}\n")
        handle.write(f"  {len(hoppings)}\n")
        degeneracies = [1] * len(hoppings)
        while degeneracies:
            chunk, degeneracies = degeneracies[:15], degeneracies[15:]
            handle.write("".join(f"{value:5d}" for value in chunk) + "\n")
        for r, matrix in hoppings:
            for i in range(n_orb):
                for j in range(n_orb):
                    handle.write(
                        f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{i + 1:5d}{j + 1:5d}"
                        f"{matrix[i, j].real:22.12f}{matrix[i, j].imag:22.12f}\n"
                    )
    with open("wannier90.win", "w") as handle:
        handle.write("num_wann = 4\nnum_bands = 4\nspinors = .true.\n\n")
        handle.write("BEGIN UNIT_CELL_CART\n")
        for row in LATTICE:
            handle.write("  " + " ".join(f"{value:.10f}" for value in row) + "\n")
        handle.write("END UNIT_CELL_CART\n\n")
        handle.write("BEGIN ATOMS_FRAC\n")
        for position in POSITIONS:
            handle.write("Fe " + " ".join(f"{value:.10f}" for value in position) + "\n")
        handle.write("END ATOMS_FRAC\n")
    with open("wannier90_centres.xyz", "w") as handle:
        handle.write("4\nWannier centres\n")
        for position in POSITIONS:
            cartesian = position @ LATTICE
            handle.write("X " + " ".join(f"{value:.6f}" for value in cartesian) + "\n")
            handle.write("X " + " ".join(f"{value:.6f}" for value in cartesian) + "\n")


if __name__ == "__main__":
    main()
