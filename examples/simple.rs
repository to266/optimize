extern crate ndarray;
extern crate optimize;

use ndarray::prelude::*;

use optimize::{Minimizer, NelderMeadBuilder};

pub fn main() {
    let function =
        |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
    let minimizer = NelderMeadBuilder::default()
        .xtol(1e-6f64)
        .ftol(1e-6f64)
        .build()
        .unwrap();
    let args = Array::from_vec(vec![3.0, -8.3]);
    let ans = minimizer.minimize(&function, args.view());
    println!("Final optimized arguments: {}", ans);
}
