use Rustb::*;
use ndarray::{arr1, array};
use num_complex::Complex;

fn main() -> Result<()> {
    // A simple cubic one-orbital model:
    // E(k) = -2t [cos(kx) + cos(ky) + cos(kz)] in lattice gauge.
    let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let orb = array![[0.0, 0.0, 0.0]];
    let mut model = Model::<false, 3>::tb_model(lat, orb, None)?;
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0, 0]), None);
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[0isize, 1, 0]), None);
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[0isize, 0, 1]), None);

    // Light incident along +z.  The Jones vector below is circular
    // polarization in the transverse (e1,e2) plane.
    let incident = IncidentBasis::from_direction(&arr1(&[0.0, 0.0, 1.0]))?;
    let circular = incident.polarization([
        Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex::new(0.0, 1.0 / 2.0_f64.sqrt()),
    ]);

    // a_complex is e A / hbar in inverse-lattice-length units.  Here the
    // lattice constant is 1, so the amplitude is dimensionless in this toy
    // model.
    let amplitude = 0.15;
    let drive = FloquetDrive::with_modes(
        0.8,
        vec![LightMode::new(1, circular.mapv(|z| amplitude * z))],
    );
    let trunc = FloquetTruncation::new(1, 128);

    println!("# kx    quasienergies in [-omega/2, omega/2)");
    for ik in 0..=8 {
        let kx = 0.5 * ik as f64 / 8.0;
        let k = arr1(&[kx, 0.0, 0.0]);
        let qe = model.floquet_quasienergy_onek(&k, &drive, &trunc, Gauge::Lattice)?;
        println!(
            "{kx:8.5}  {}",
            qe.iter()
                .map(|x| format!("{x:12.7}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    Ok(())
}
