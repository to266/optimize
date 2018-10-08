extern crate optimize;
extern crate ndarray;

use optimize::vector::NelderMead;
use ndarray::prelude::*;

/// This implementation of bounded-step ensures that only vectors on the standard simplex are considered,
/// i.e. all elements are positive and sum up to 1.
#[inline]
fn bounded_step(max_growth: f64, direction: ArrayView1<f64>, from: ArrayView1<f64>) -> Array1<f64> {
        let delta = direction.iter().zip(&from)
        .map(|(d,f)| if *d<0.0 {f/d.abs()} else {max_growth})
        .fold(max_growth, |acc, x| if acc < x {acc} else {x});

        let vec = (&from + &(delta * &direction)).mapv(|xi| 0f64.max(xi));
        let s = vec.scalar_sum();
        vec / s
}

fn main() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        nm.bounded_step = &bounded_step;
        let n = 5;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();  

        // Here we choose as initial vectors a simplex on the standard n-simplex.
        // This ensures that searching only takes place on the standard n-simplex.
        let x0 = Array1::ones(n) / n as f64;
        let eps = 1f64 / n as f64;
        let x: Array2<f64> = Array::eye(x0.len()) * eps + &x0 * (1.0-eps);
        nm.minimize(&f, x);

        println!("{:?}", x0);
}