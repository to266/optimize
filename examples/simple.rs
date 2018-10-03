extern crate optimize;
extern crate ndarray;

use optimize::vector::NelderMead;
use ndarray::prelude::*;

fn main() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        let n = 5;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();        
        let mut x0 = Array1::ones(n) / n as f64;
        nm.minimize(&f, x0.view_mut());

        println!("{:?}", x0);
}