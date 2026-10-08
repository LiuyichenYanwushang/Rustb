use ndarray::*;
use num_traits::Zero;
use std::fmt::Display;

/// Write a numeric matrix with sign-aligned whitespace.
pub fn write_txt<T: Display + PartialOrd + Zero>(
    data: &Array2<T>,
    output: &str,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(output)?;
    let n = data.len_of(Axis(0));
    let s = data.len_of(Axis(1));
    let mut s0 = String::new();
    for i in 0..n {
        for j in 0..s {
            if data[[i, j]] >= T::zero() {
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

/// Write one numeric value per line, with sign-aligned whitespace.
pub fn write_txt_1<T: Display + PartialOrd + Zero>(
    data: &Array1<T>,
    output: &str,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(output)?;
    let n = data.len_of(Axis(0));
    let mut s0 = String::new();
    for i in 0..n {
        if data[[i]] >= T::zero() {
            s0.push_str(" ");
        }
        let aa = format!("{:.6}\n", data[[i]]);
        s0.push_str(&aa);
    }
    writeln!(file, "{}", s0)?;
    Ok(())
}
