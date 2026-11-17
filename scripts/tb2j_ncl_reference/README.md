# TB2J non-collinear reference

`gen_reference.py` writes the Wannier90 files of a two-site spinor model whose
local exchange fields point along different axes (`z` on site 0, `x` on site 1),
so the exchange acquires a finite Dzyaloshinskii–Moriya vector. Running TB2J's
spinor path on it gives the reference values quoted in the script's docstring.

```bash
cd $(mktemp -d)
python3 /path/to/gen_reference.py
wann2J.py --path . --spinor --prefix_spinor wannier90 --groupby spin \
    --elements Fe --efermi 0.0 --nz 60 --kmesh 6 6 1
sed -n '/Exchange:/,$p' TB2J_results/exchange.out
```

Two parsing traps, both discovered the hard way: `wannier90_centres.xyz` must
use the symbol `X` for the centres (HamiltonIO keeps only `X` entries), and the
`.win` must carry the `BEGIN UNIT_CELL_CART` / `BEGIN ATOMS_FRAC` blocks — a
plain `lattice = ...` line is rejected.

The `J_iso`, `DMI` and `J_ani` that TB2J prints are expressed in each site's
local spin frame, so a Rust implementation of the `A^{uv}` extraction must apply
the same frame rotation before it can be compared against these numbers.

## Calibration notes (what the next attempt should not re-learn)

1. **Spinor indexing is spin-major**: in `Model<true, ..>` the state index of
   (orbital `o`, spin `s`) is `s * n_orb + o`, not `2 * o + s`. Reading the
   on-site block with the latter silently mixes the two sites and produces a
   meaningless `P_i` (verified: with the correct stride, site 0 gives
   `P = 1.0` along `z` and site 1 `P = 1.0` along `x` for a field splitting of
   1 eV).
2. **`P_i` is the full splitting, not half of it**: `Tr(sigma^z H_onsite)`
   equals `H_up - H_dn`, i.e. `m`. Consequently
   `A^{00} - A^{11} - A^{22} - A^{33} = 4 Tr[P G^up P G^dn]`, whereas the
   collinear path accumulates `Delta^2 Tr[G^up G^dn]` with the same
   `Delta = m`. A spinor implementation that reuses the collinear
   `contour_factor / (4 pi)` therefore needs an extra `1/4` (and TB2J's
   `J_iso` is the *trace* over the three spin components, i.e. three times the
   per-component coupling).
3. A first implementation reproduced `P_i` and the Pauli projections but
   returned a tensor of order `1e-9 eV`, i.e. essentially zero. The likely
   culprit is the contour range: `ExchangeParams` defaults to a wide
   `emin`/`emax`, while TB2J derives them from the band edges of this small
   model (bands live near `+-0.5 eV`). Check the contour bounds before
   suspecting the `A^{uv}` algebra.
