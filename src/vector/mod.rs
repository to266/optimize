mod nelder_mead;
pub use self::nelder_mead::NelderMead;

use ndarray::{ArrayView1, Array1};

#[allow(dead_code)]
fn grad<'a>(f: &Fn(ArrayView1<f64>)->f64, x: ArrayView1<f64>) -> Array1<f64> {
    let eps = 1e-8;
    let f0 = f(x);
    let mut dir = x.to_owned();
    let mut grad = x.to_owned();
    for i in 0..x.len() {
        dir[i] += eps;
        grad[i] = ( f(dir.view()) - f0 ) / eps;
        dir[i] -= eps;
    }
    grad
}