use ndarray::{ArrayView1, Array1};
use ::Status;  
use vector::{Bound, Target, Gradient, unbounded_step, approximate_gradient_neg};
use scalar::GoldenRatio;

pub struct LineSearch<'a> {
    pub ulps: i64,
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub bounded_step: &'a Bound<'a>,
    pub direction: &'a Gradient<'a>,
}


impl<'a> LineSearch<'a> {
    pub fn new() -> Self {
        LineSearch {
            ulps: 1,
            max_iter: 5000,
            ftol: 1e-9,
            xtol: 1e-9,
            bounded_step: &unbounded_step,
            direction: &approximate_gradient_neg,
        }
    }

    pub fn minimize(&self, f: &Target, x0: ArrayView1<f64>) -> (Array1<f64>, Status) {
        let mut status = Status::NotFinished;
        let mut x = x0.to_owned();
        let mut f0 = f(x0);
        let mut iter = 0;
        while status == Status::NotFinished {
            iter += 1;
            let dir = (self.direction)(f, x.view());
            let step = self.iterate(f, x.view(), dir.view());
            x += &(step*&dir);
            let f1 = f(x.view());
            status = self.update_status(iter, f0, f1, step + dir.mapv(f64::abs).scalar_sum());
            println!("{}\t{}\t{:?}", f0, f1, status);
            f0 = f1;
        }

        (x, status)
    }

    fn iterate(&self, f: &Target, x0: ArrayView1<f64>, dir: ArrayView1<f64>) -> f64 {
        let mut fprev = f(x0);
        let mut stepsize = (self.bounded_step)(1.0, dir, x0);
        let fun = |x| f((&x0 + &(x * &dir)).view());
        let mut fstepped = fun(stepsize);

        // find a bracketing interval; double the interval until f stops improving
        while fstepped < fprev {
            stepsize = (self.bounded_step)(2.0 * stepsize, dir, x0);
            fprev = fstepped;
            fstepped = fun(stepsize);
        } 

        let mut gr = GoldenRatio::new();
        gr.ftol = self.ftol;
        gr.xtol = self.xtol;
        gr.minimize(&fun, 0.0, stepsize).0
    }

    #[inline]
    fn update_status(&self, iter: usize, f0: f64, f1: f64, xdiff: f64) -> Status {
        if (f1 - f0).abs() < self.ftol {
            return Status::FtolConvergence
        } else if xdiff < self.xtol {
            return Status::XtolConvergence
        } else if iter > self.max_iter {
            return Status::MaxIterReached
        } else {
            return Status::NotFinished
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_unbounded() {
        let n = 5;
        let ls = LineSearch::new();
        let f = |x: ArrayView1<f64>| (&x - 0.5).mapv(|xi| xi*xi).scalar_sum();
        let (xout, status) = ls.minimize(&f, Array::ones(n).view());

        println!("{:?}", status);
        println!("{:?}", xout);
        assert!(xout.all_close(&(Array::ones(n)/2.0 as f64), 1e-5));
    }
    #[test]
    fn test_bounded_on_the_simplex() {
        let n = 10;
        let dir = |f: &Target, x: ArrayView1<f64>| {
            let mut g = approximate_gradient_neg(f, x);
            // project the gradient on the simplex:
            let mask = x.iter().zip(&g).map(|(xi, gi)| if *xi > 0.0 || *gi >= 0.0 {1.0} else {0.0}).collect::<Array1<f64>>();
            g *= &mask; // set the directions at borders pointing outside the border to 0
            g -= &(g.scalar_sum() * &mask / mask.scalar_sum()); // projection onto the simplex
            g
        };
        let bound = |max_step: f64, x: ArrayView1<f64>, dir: ArrayView1<f64>| {
            dir.iter().zip(&x)
            .map(|(d,f)| if *d<0.0 {f/d.abs()} else {max_step})
            .fold(max_step, |acc, x| if acc < x {acc} else {x})
        };
        let f = |x: ArrayView1<f64>| {
            x.mapv(|xi| xi*xi).scalar_sum()
        };
        let mut x0 = Array1::zeros(n);
        x0[0] = 1.0;

        let mut ls = LineSearch::new();
        ls.direction = &dir;
        ls.bounded_step = &bound;
        let (xout, status) = ls.minimize(&f, x0.view());
        println!("{:?}", status);
        println!("{:?}", xout);
    }
}