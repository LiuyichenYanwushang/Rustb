This crate is combined with wannier90's wannier model or tight-binding model to calculate various physical properties including:
- Conductivity calculations
- Energy band structures
- Density of states
- Edge states and surface Green's functions
- Wilson loops and topological invariants
- File I/O utilities for data export

## Key Features

- **Band Structure**: Eigenvalue problem solutions
- **Transport Properties**: Anomalous Hall, spin Hall, and nonlinear conductivities
- **Topological Calculations**: Chern numbers, Berry curvature, Wannier centers
- **File I/O**: Utilities for writing 1D and 2D arrays to formatted text files

## Dependencies

Using this crate requires:
- `num-complex` for complex number support
- `ndarray` for multi-dimensional arrays
- `ndarray-linalg` for linear algebra operations

For optimal performance with `ndarray-linalg`, enable features like "intel-mkl-static" or "openblas-static". See https://github.com/rust-ndarray/ndarray-linalg for details.

## File I/O Utilities

The library provides file output functions:
- `write_txt`: Export 2D arrays to formatted text files
- `write_txt_1`: Export 1D arrays to formatted text files

These utilities handle proper number formatting and spacing for scientific data analysis.

