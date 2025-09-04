use std::f64::consts::PI;
pub fn gauss(x: f64, eta: f64) -> f64 {
    //高斯函数
    let a = (x / eta);
    let g = (-a * a / 2.0).exp();
    let g = 1.0 / (2.0 * PI).sqrt() / eta * g;
    g
}
