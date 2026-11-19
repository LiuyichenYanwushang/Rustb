//! Extract TB2J-style exchange parameters from a Wannier90 up/down HR pair.
//!
//! ```text
//! cargo run --release --features openblas-system --example mgfe6ge6_exchange -- \
//!     <dir with up/ and dn/> [kmesh kx ky kz] [nz]
//! ```
//!
//! Both HR files are expected to be shifted by the same amount so that
//! `E_F = 0`. The third argument to `Model::from_hr` is the energy to subtract
//! on import; it is `0.0` here because the files are already shifted.

use Rustb::magnon::{ExchangeParams, Magnon};
use Rustb::ndarray_lapack::eigh_x;
use Rustb::wannier90::Wannier90;
use Rustb::{AtomType, Model};
use ndarray::{Array1, Array2};
use ndarray_linalg::UPLO;

fn main() -> Rustb::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: <dir with up/ dn/> [kx ky kz] [nz]");
    let mut kmesh = Array1::from_vec(vec![9usize, 9, 5]);
    let mut nz = 100usize;
    let mut method = String::from("cfr");
    let rest: Vec<String> = args.collect();
    if rest.len() >= 3 {
        for (slot, value) in kmesh.iter_mut().zip(rest.iter()) {
            *slot = value.parse().expect("kmesh components must be integers");
        }
    }
    if rest.len() >= 4 {
        nz = rest[3].parse().expect("nz must be an integer");
    }
    if rest.len() >= 5 {
        method = rest[4].clone();
    }

    let base = if dir.ends_with('/') {
        dir.clone()
    } else {
        format!("{dir}/")
    };
    println!("reading {base}up and {base}dn ...");
    let up: Model<false, 3> = Model::from_hr(&format!("{base}up/"), "wannier90", 0.0)?;
    let down: Model<false, 3> = Model::from_hr(&format!("{base}dn/"), "wannier90", 0.0)?;
    println!(
        "up: norb={} natom={} nR={} | dn: norb={} natom={} nR={}",
        up.norb(),
        up.natom(),
        up.hamR.nrows(),
        down.norb(),
        down.natom(),
        down.hamR.nrows()
    );

    let fe: Vec<usize> = up
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.atom_type() == AtomType::Fe)
        .map(|(index, _)| index)
        .collect();
    println!("Fe atoms: {fe:?}");
    for &index in &fe {
        println!(
            "  atom {index}: orbitals {:?}",
            up.atoms[index]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        );
    }

    let moments = site_moments(&up, &down, &fe, &kmesh)?;
    println!("Wannier magnetic moments (mu_B):");
    for (index, value) in fe.iter().zip(moments.iter()) {
        println!("  Fe {index}: {value:+.4}");
    }
    let _ = &moments;
    // The moments printed above are what the library derives internally from
    // the two channels, so they are not passed in again.
    let mut params = ExchangeParams::new(&kmesh)
        .with_nz(nz)
        .with_magnetic_atoms(fe.clone());
    if method.eq_ignore_ascii_case("semi") {
        params = params.with_semicircle_contour();
    } else if method.eq_ignore_ascii_case("cfr") {
        params = params.with_cfr_contour(600.0);
    } else {
        panic!("unknown contour method '{method}', use 'cfr' or 'semi'");
    }

    let started = std::time::Instant::now();
    let magnon = Magnon::<3>::from_collinear(&up, &down, &params)?;
    println!(
        "extraction finished in {:.1} s: nspin={} nR={}",
        started.elapsed().as_secs_f64(),
        magnon.nspin(),
        magnon.nR()
    );

    // Collect directed Fe-Fe pairs and group them by distance.
    let n = magnon.nspin();
    let mut pairs: Vec<(f64, usize, usize, [isize; 3], f64)> = Vec::new();
    for i_r in 0..magnon.nR() {
        let translation: [f64; 3] = [0, 1, 2].map(|axis| magnon.hamR[[i_r, axis]] as f64);
        for i in 0..n {
            for j in 0..n {
                if i == j && translation.iter().all(|&value| value == 0.0) {
                    continue;
                }
                let fractional: [f64; 3] = [0, 1, 2]
                    .map(|axis| translation[axis] + magnon.orb[[j, axis]] - magnon.orb[[i, axis]]);
                let cartesian: [f64; 3] = [0, 1, 2].map(|component| {
                    (0..3)
                        .map(|axis| fractional[axis] * magnon.lat[[axis, component]])
                        .sum()
                });
                let distance = (cartesian[0] * cartesian[0]
                    + cartesian[1] * cartesian[1]
                    + cartesian[2] * cartesian[2])
                    .sqrt();
                // Isotropic coupling = trace of the 3x3 tensor / 3, in meV.
                let tensor = magnon.jtensor(i, j, i_r)?;
                let j_iso = (0..3)
                    .map(|component| tensor[component][component])
                    .sum::<f64>()
                    / 3.0
                    * 1000.0;
                if j_iso.is_finite() && j_iso.abs() > 1e-6 {
                    pairs.push((
                        distance,
                        i,
                        j,
                        magnon
                            .hamR
                            .row(i_r)
                            .to_owned()
                            .into_raw_vec_and_offset()
                            .0
                            .try_into()
                            .unwrap(),
                        j_iso,
                    ));
                }
            }
        }
    }
    println!("{} nonzero exchange entries", pairs.len());

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut index = 0;
    println!("#  d(A)     count   J_mean(meV)  J_min      J_max     spread");
    while index < pairs.len() {
        let start = index;
        while index < pairs.len() && (pairs[index].0 - pairs[start].0).abs() < 5e-3 {
            index += 1;
        }
        let shell = &pairs[start..index];
        let count = shell.len();
        let mean = shell.iter().map(|entry| entry.4).sum::<f64>() / count as f64;
        let min = shell
            .iter()
            .map(|entry| entry.4)
            .fold(f64::INFINITY, f64::min);
        let max = shell
            .iter()
            .map(|entry| entry.4)
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:7.3} {:6}  {:+.6e}  {:+.6e}  {:+.6e}  {:.3e}",
            shell[0].0,
            count,
            mean,
            min,
            max,
            max - min
        );
    }

    // Reference values from TB2J (`exchange.out`, efermi = 0, kmesh 9 9 4).
    println!("\nTB2J reference: J1 = +24.7825 meV (24 bonds, 2.631 A),");
    println!("                J(4.556 A) = -0.6867 meV, J(5.064 A) = +1.1599 meV");
    Ok(())
}

/// Magnetic moment of each site from the two channel density matrices at T = 0.
fn site_moments(
    up: &Model<false, 3>,
    down: &Model<false, 3>,
    sites: &[usize],
    mesh: &Array1<usize>,
) -> Rustb::Result<Vec<f64>> {
    let n = up.norb();
    let points: Array2<f64> = Rustb::kpoints::gen_kmesh::<f64>(mesh)?;
    let mut up_occ = Array2::<f64>::zeros((n, n));
    let mut down_occ = Array2::<f64>::zeros((n, n));
    for k in points.axis_iter(ndarray::Axis(0)) {
        for (model, target) in [(&up, &mut up_occ), (&down, &mut down_occ)] {
            let hk = model.gen_ham(&k, Rustb::Gauge::Lattice);
            let (values, vectors) =
                eigh_x(&hk, (f64::MIN / 2.0, f64::MAX / 2.0), 0.0, UPLO::Lower)?;
            for band in 0..n {
                if values[band] < 0.0 {
                    for alpha in 0..n {
                        let left = vectors[[band, alpha]];
                        for beta in 0..n {
                            target[[alpha, beta]] += (left * vectors[[band, beta]].conj()).re;
                        }
                    }
                }
            }
        }
    }
    let scale = 1.0 / points.nrows() as f64;
    let mut moments = Vec::with_capacity(sites.len());
    for &atom in sites {
        let orbitals = up.atoms[atom].orbitals();
        let mut total = 0.0;
        for id in orbitals {
            let index = id.index();
            total += (up_occ[[index, index]] - down_occ[[index, index]]) * scale;
        }
        moments.push(total);
    }
    Ok(moments)
}
