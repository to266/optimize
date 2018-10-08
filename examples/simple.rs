extern crate optimize;
#[macro_use]
extern crate ndarray;

use optimize::vector::NelderMead;
use ndarray::prelude::*;


fn main() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        let n = 5;
        let f = |x: ArrayView1<f64>| x.mapv(f64::abs).scalar_sum();        

        let mut x: Array2<f64> = Array2::zeros((n+1, n));
        x.slice_mut(s![1.., ..]).assign(&Array::eye(n));
        let (status, out) = nm.minimize(&f, x);

        println!("{:?}", status);
        println!("{:?}", out);
}