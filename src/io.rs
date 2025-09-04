use ndarray::*;
use crate::usefloat;
pub fn write_txt<T: usefloat>(data: &Array2<T>, output: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(output).expect("Unable to BAND.dat");
    let n = data.len_of(Axis(0));
    let s = data.len_of(Axis(1));
    let mut s0 = String::new();
    for i in 0..n {
        for j in 0..s {
            if data[[i, j]] >= T::from(0.0) {
                s0.push_str("     ");
            } else {
                s0.push_str("    ");
            }
            let aa = format!("{:.6}", data[[i, j]]);
            s0.push_str(&aa);
        }
        s0.push_str("\n");
    }
    writeln!(file, "{}", s0)?;
    Ok(())
}

pub fn write_txt_1<T: usefloat>(data: &Array1<T>, output: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(output).expect("Unable to BAND.dat");
    let n = data.len_of(Axis(0));
    let mut s0 = String::new();
    for i in 0..n {
        if data[[i]] >= T::from(0.0) {
            s0.push_str(" ");
        }
        let aa = format!("{:.6}\n", data[[i]]);
        s0.push_str(&aa);
    }
    writeln!(file, "{}", s0)?;
    Ok(())
}
