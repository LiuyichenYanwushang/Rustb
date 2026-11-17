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
