#[macro_use(s)]
extern crate ndarray;

extern crate num_traits;
extern crate float_cmp;

use ndarray::prelude::*;
use std::fmt::Debug;
use float_cmp::ApproxEqUlps;

pub struct Minimizer {}

impl Minimizer {
    pub fn minimize<F, R>(&self, func: F, args: ArrayView1<f64>)
    where
        F: Fn(ArrayView1<f64>) -> R,
        R: Debug,
    {
        let ans = func(args);
        println!("calculated this: {:?}", ans);
        self.minimize_neldermead(func, args);
    }

    fn minimize_neldermead<F, R>(&self, func: F, args: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> R,
        R: Debug,
    {
        let mut x0 = args.to_owned();
        let N: usize = x0.len();
        let mut sim = Array2::<f64>::zeros((N + 1, N));
        println!("Sim at zero: \n{}", sim);
        let nonzdelt = 0.05;
        let zdelt = 0.00025;
        sim.slice_mut(s![0, ..]).assign(&x0);
        for k in 0..N {
            let mut y = x0.clone();
            y[k] = match y[k].approx_eq_ulps(&0., 2) {
                true => zdelt,
                _ => (1. + nonzdelt) * y[k],
            };
            // sim[k + 1] = y
            sim.slice_mut(s![k+1, ..]).assign(&y);
        }
        println!("sim initialized:\n{}", sim);

        x0
    }

    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
