# Rustb

A comprehensive Rust library for tight-binding model calculations in condensed matter physics.

[![Crates.io](https://img.shields.io/crates/v/Rustb.svg)](https://crates.io/crates/Rustb)

## Overview

Rustb computes electronic band structures, density of states, transport properties (anomalous Hall, spin Hall, nonlinear conductivities), topological invariants (Chern numbers, Wilson loops, Wannier centers), and surface states via iterative Green's functions. It supports both **analytical tight-binding models** and **Wannier90** Hamiltonians, as well as **Slater-Koster** parameterized models.

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
Rustb = "0.6"
ndarray = "0.17"
num-complex = "0.4"
gnuplot = "0.0.39"     # for plotting
```

### Minimal Graphene Example

```rust
use Rustb::*;
use ndarray::*;
use ndarray::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

fn main() {
    // Graphene lattice: two triangular sublattices A and B
    // Lattice vectors (rows): a₁ = (√3, -1)a₀, a₂ = (√3, 1)a₀
    let a0 = 1.0;
    let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]) * a0;

    // Orbital positions (fractional): A at origin, B at (1/3, 1/3)
    let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);

    // Create spinless 2D model, dim_r=2, norb=2
    let mut model = Model::tb_model(2, lat, orb, false, None).unwrap();

    // Nearest-neighbor hopping t ≈ -2.85 eV (graphene)
    let t = Complex::new(-2.85, 0.0);
    model.add_hop(t, 0, 1, &array![0, 0], SpinDirection::None);   // A→B in home cell
    model.add_hop(t, 0, 1, &array![-1, 0], SpinDirection::None);  // A→B in x-left cell
    model.add_hop(t, 0, 1, &array![0, -1], SpinDirection::None);  // A→B in y-left cell

    // k-path: Γ → K → K' → Γ
    let path = array![
        [0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [0.0, 0.0]
    ];
    let label = vec!["Γ", "K", "K'", "Γ"];
    let nk = 1001;

    // Generate band structure plot
    model.show_band(&path, &label, nk, "./");
}
```

### Adding Spin-Orbit Coupling (Kane-Mele Model)

```rust
// Spinful model: each orbital is doubled (↑, ↓)
let mut model = Model::tb_model(2, lat, orb, true, None).unwrap();

// On-site staggered potential
model.set_onsite(&arr1(&[0.5, -0.5]), SpinDirection::None);

// Spin-orbit coupling: i·λ_SO·ν_ij·σ_z  (ν_ij = ±1 for clockwise/counterclockwise)
let soc = Complex::new(0.0, 0.2);
model.add_hop(soc, 0, 0, &array![1, 0], SpinDirection::z);      // + for A→A
model.add_hop(-soc, 1, 1, &array![1, 0], SpinDirection::z);     // - for B→B
model.add_hop(soc, 0, 0, &array![0, 1], SpinDirection::z);
model.add_hop(-soc, 1, 1, &array![0, 1], SpinDirection::z);
model.add_hop(soc, 0, 0, &array![1, -1], SpinDirection::z);
model.add_hop(-soc, 1, 1, &array![1, -1], SpinDirection::z);
```

## Features

### Band Structure & Density of States

```rust
// Band structure along a k-path
let (kvec, kdist, knode) = model.k_path(&path, nk).unwrap();
let eval = model.solve_band_all_parallel(&kvec);  // parallelized over k-points
model.show_band(&path, &label, nk, "./output/").unwrap();

// Density of states with Gaussian broadening
let kmesh = arr1(&[100, 100]);
let (energy, dos) = model.dos(&kmesh, -6.0, 6.0, 500, 0.05).unwrap();
model.show_dos(&energy, &dos, "./output/");
```

### Topological Properties

```rust
// Chern number via Berry curvature integration
let kmesh = arr1(&[101, 101]);
let (chern, _) = model.chern_number(&kmesh, 0.0, 0usize, 1e-4).unwrap();
println!("Chern number = {}", chern);

// Wilson loop / Wannier charge centers
let (wcc, _) = model.wilson_loop(&kmesh, None, None, 0.0, 0).unwrap();
model.show_wcc(&wcc, "./output/");

// Berry curvature at each k-point
let berry = model.berry_curvature(&kvec, &dir_1, &dir_2, 300.0, 0.0, 0.0, 0, 1e-4).unwrap();
```

### Transport Properties

```rust
// Anomalous Hall conductivity σ_xy(μ, T)
let mu = Array1::linspace(-3.0, 3.0, 200);
let sigma = model.Hall_conductivity(&kmesh, &mu, 300.0, 0.0, 0, None, 1e-4).unwrap();

// Nonlinear Hall conductivity (Berry curvature dipole)
let dir1 = arr1(&[1.0, 0.0]);
let dir2 = arr1(&[0.0, 1.0]);
let dir3 = arr1(&[1.0, 0.0]);
let sigma_nl = model.Nonlinear_Hall_conductivity_Intrinsic(
    &kmesh, &dir1, &dir2, &dir3, &mu, 300.0, 0.0, 0, 1e-4
).unwrap();
```

### Surface States (Iterative Green's Function)

```rust
// Semi-infinite surface along direction 0, surface normal direction 1
let mut surf = surf_Green::new(&model, 0, 1, 1e-8, 100000).unwrap();

// Surface spectral function A(k_∥, ω)
let k_para = arr1(&[0.5, 0.0]);  // momentum parallel to surface
let omega = Array1::linspace(-4.0, 4.0, 500);
let (A_surf, A_bulk) = surf.spectral_function(&k_para, &omega, 0.01).unwrap();
```

### Magnetic Field (Peierls Substitution)

```rust
// Uniform magnetic field out-of-plane, supercell expansion [m, n], N_ϕ flux quanta
let model_b = model.add_magnetic_field(2, [10, 10], 1).unwrap();

// Hofstadter butterfly: scan flux per primitive cell
let phi_list = Array1::linspace(0.0, 1.0, 101);
let spectrum = model.hofstadter_butterfly(&phi_list, &[10, 10], 2).unwrap();
```

### Band Unfolding (for supercells)

```rust
// Unfold supercell band structure back to primitive cell BZ
let U = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
let super_model = model.make_supercell(&U).unwrap();
let A_unfolded = super_model.unfold(&U, &path, nk, -3.0, 5.0, nk, 1e-2, 1e-3).unwrap();
draw_heatmap(&A_unfolded, "./unfold_band.pdf");
```

### Supercell & Geometry

```rust
// Create a supercell: new_lat = U · old_lat
let U = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
let super_model = model.make_supercell(&U).unwrap();

// Cut a slab/nanoribbon: remove bonds crossing the cut plane
let plane_normal = arr1(&[0.0, 1.0]);  // cut along y direction
let slab = model.cut_piece(&plane_normal, 10.0).unwrap();
```

### Slater-Koster Models

```rust
use Rustb::*;

// Define SK parameters (two-center integrals)
let params = SkParams::new()
    .with_onsite("s", "A", 0.0)
    .with_onsite("s", "B", 0.5)
    .with_hopping("s", "s", "Vssσ", -2.0)
    .with_hopping("s", "p", "Vspσ", 1.5)
    .with_hopping("p", "p", "Vppσ", 3.0)
    .with_hopping("p", "p", "Vppπ", -0.5);

// Build atoms
let atoms = vec![
    SkAtom::new("A", &[0.0, 0.0, 0.0], &["s"]),
    SkAtom::new("B", &[0.5, 0.5, 0.0], &["s", "p"]),
];

// Create SK model and convert to TB model
let sk = SlaterKosterModel::new(lat, atoms, params);
let model = sk.to_tb_model().unwrap();
```

### Wannier90 Interface

```rust
// Read Wannier90 _hr.dat, _r.dat, _wsvec.dat files
let model = Model::from_wannier90("wannier90_hr.dat", "wannier90_r.dat",
                                   "wannier90_wsvec.dat", true).unwrap();

// Write Wannier90 format files from a model
model.output_hr("output_hr.dat").unwrap();
```

## Model Construction API

### Creating a Model

```rust
// Spinless model, dim=2 or 3
let model = Model::tb_model(dim, lat, orb, false, None)?;

// Spinful model
let model = Model::tb_model(dim, lat, orb, true, None)?;

// With custom orbital projections
let model = Model::tb_model(dim, lat, orb, spin, Some(&orb_proj))?;
```

### Adding Hopping & On-site Terms

```rust
// add_hop: hermitian conjugate at -R is added automatically
model.add_hop(t, orb_i, orb_j, &R_vec, spin_dir);

// set_hop: manually set without automatic hermitian conjugate
model.set_hop(t, orb_i, orb_j, &R_vec, spin_dir);

// Bulk-set all orbitals with same value
model.add_onsite(&values, spin_dir);   // adds to existing
model.set_onsite(&values, spin_dir);   // overwrites
```

Where `spin_dir` controls the spin structure:
- `SpinDirection::None` → identity in spin space
- `SpinDirection::x / y / z` → Pauli matrix σ_x / σ_y / σ_z

### Gauge Choice

```rust
// Atomic gauge: orbital positions included in Bloch phase
let (v, hamk) = model.gen_v(&kvec, Gauge::Atom);

// Lattice gauge: only R vectors in Bloch phase
let (v, hamk) = model.gen_v(&kvec, Gauge::Lattice);
```

## BLAS/LAPACK Backend

For optimal performance, enable a BLAS backend:

```toml
[dependencies]
Rustb = { version = "0.6", features = ["intel-mkl-static"] }
# or: "intel-mkl-system", "openblas-static", "openblas-system", "netlib-static", "netlib-system"
```

Without features, `ndarray-linalg` uses system BLAS (may be slow). Intel MKL provides the best performance on x86_64.

## Physical Constants

```rust
use Rustb::phy_const::*;
// ħ, e, k_B, μ_B, Φ_0, etc.
```

Constants in SI-based eV·Å units:
- `HBAR_EV_S` = 6.582119569e-16 eV·s
- `HBAR2_OVER_2M_EV_A2` = 3.81 eV·Å²
- `MU_B_EV_PER_T` = 5.7883818060e-5 eV/T
- `FLUX_QUANTUM_T_M2` = 4.135667696e-15 T·m²
- `K_B_EV_PER_K` = 8.617333262145e-5 eV/K

## Plotting

The library uses `gnuplot` with `pdfcairo` terminal. Install gnuplot:

```bash
# Ubuntu/Debian
sudo apt install gnuplot

# macOS
brew install gnuplot
```

Built-in plotting functions:
- `model.show_band(&path, &label, nk, dir)` — band structure
- `model.show_dos(&energy, &dos, dir)` — density of states
- `model.show_surf_state(&surf_green, &omega, &k_para, dir)` — surface spectral function
- `model.show_wcc(&wcc, dir)` — Wilson loop spectrum
- `draw_heatmap(&data, filename)` — 2D heatmap

## License

MIT OR Apache-2.0
