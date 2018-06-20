extern crate optimize;

use optimize::Minimizer;

pub fn main() {
    let function = |x: &[f64]| {
        (1.0 - x[0]).powi(2) + 100.0*(x[1] - x[0].powi(2)).powi(2)
    };
    let minimizer = Minimizer::new();
    minimizer.minimize(&function, &vec![-3.0, -4.0]);
}
