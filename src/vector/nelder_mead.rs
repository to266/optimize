//! TODO explain Nelder-mead a bit here. Especialy the convex search region.

use ndarray::{ArrayView1, Array1, Array2};
use float_cmp::{ApproxOrdUlps};
// use ::Status;
use Status;
use vector::{Bound, Target, unbounded_step};

pub struct NelderMead<'a> {
    pub ulps: i64,
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub alpha: f64, // > 0; reflection multiplier
    pub gamma: f64, // > 1; expansion multiplier
    pub rho: f64, // > 0, < 0.5; contraction factor
    pub sigma: f64, // > 0, < 1; shrinkage factor
    pub bounded_step: &'a Bound<'a>// &BoundedStep
}

/// bounded_step checks if the simplex does not walk outside of the convex search region. 
/// To enforce convex constraints, implement your own version of bounded_step where you 
/// limit the stepsize in <direction>.
// fn unbounded_step(max_growth: f64, direction: ArrayView1<f64>, from: ArrayView1<f64>) -> f64 {
//     &from + &(max_growth * &direction)
// }

type Simplex = Vec<(f64, Array1<f64>)>;
// type Target<'a> = Fn(ArrayView1<f64>)->f64 + 'a;
// type Bound<'a> = Fn(f64, ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + 'a;

impl<'a> NelderMead<'a> {
    
    pub fn new() -> Self {
        NelderMead {
            ulps: 1,
            max_iter: 1000,
            ftol: 1e-9,
            xtol: 1e-9,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            bounded_step: &unbounded_step
        }
    }

    pub fn minimize(&mut self, f: &Target, init_simplex: Array2<f64>) -> (Status, Array1<f64>) {
        let mut status = Status::NotFinished;
        let n = init_simplex.shape()[1];

        // initialize 
        let mut simplex = init_simplex.outer_iter().map(|xi| (f(xi), xi.to_owned())).collect::<Simplex>();
        self.order_simplex(&mut simplex);
        let mut centroid = self.centroid(&simplex);
        let mut iter = 0;

        while status == Status::NotFinished {
            iter += 1;
            let f_worst = simplex[n-1].0;
            let f_best = simplex[0].0;
            eprint!("\r{}: {}", iter, f_best);

            let reflect = &centroid - &simplex[n-1].1;
            let reflect_bound = (self.bounded_step)(self.alpha, reflect.view(), centroid.view());
            let reflected = &centroid + &(reflect_bound * &reflect);
            let f_reflected = f(reflected.view());

            if f_reflected < f_worst && f_reflected > f_best { // try reflecting the worst point through the centroid
                self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
            } else if f_reflected < f_best { // try expanding beyond the centroid
                let expand = &centroid - &reflected;
                let expand_bound = (self.bounded_step)(self.gamma, expand.view(), centroid.view());
                let expanded = &centroid + &(expand_bound * &expand);
                let f_expanded = f(expanded.view());

                if f_expanded < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, expanded, f_expanded);
                } else {
                    self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
                }
            } else { // try a contraction towards the centroid
                let contracted = &centroid - &(self.rho * (&centroid - &simplex[n-1].1));
                let f_contracted = f(contracted.view());

                if f_contracted < f_worst {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else { // shrink if all else fails
                    self.shrink(&mut simplex, f);
                    self.order_simplex(&mut simplex);
                    centroid = self.centroid(&mut simplex);
                }
            }        
            status = self.update_status(iter, &simplex);
        }
        (status, simplex.remove(0).1)
    }

    /// Update the centroid effiently, knowing only one value changed.
    /// The pattern-defeating sort of order_simplex is allready efficient
    /// given that we inserted a single out-of-place value in a sorted vec.
    #[inline]
    fn lean_update(&self, simplex: &mut Simplex, centroid: &mut Array1<f64>, xnew: Array1<f64>, fnew: f64) {
        let n = simplex.len();
        *centroid += &(&xnew / (n-1) as f64);
        simplex[n-1] = (fnew, xnew);
        self.order_simplex(simplex);
        *centroid -= &(&simplex[n-1].1 / (n-1) as f64);
    }

    /// shrink all points towards the best point.
    /// Assumes the simplex is ordered.
    #[inline]
    fn shrink(&self, simplex: &mut Simplex, f:&Target) {
        let mut iter = simplex.iter_mut();
        let (_, x0) = iter.next().unwrap();
        for (fi, xi) in iter {
            *xi *= self.sigma;
            *xi += &((1.0-self.sigma) * &x0.view());
            *fi = f(xi.view());
        }
    }

    /// calculate the centroid of all points but the worst one.
    /// Assumes that the simplex is ordered.
    #[inline]
    fn centroid(&self, simplex: &Simplex) -> Array1<f64> {
        let n = simplex.len();
        let mut centroid = Array1::zeros(simplex[0].1.len());
        for (_, xi) in simplex.iter().take(n-1) {
            centroid += xi;
        }
        centroid / (n-1) as f64
    }

    /// Gather the program status
    #[inline]
    fn update_status(&self, iter: usize, simplex: &Simplex) -> Status {
        let n = simplex.len();
        if simplex[n-1].0 - simplex[0].0 < self.ftol {
            return Status::FtolConvergence
        } else if (&simplex[n-1].1 - &simplex[0].1).mapv(f64::abs).scalar_sum() < self.xtol {
            return Status::XtolConvergence
        } else if iter > self.max_iter {
            return Status::MaxIterReached
        } else {
            return Status::NotFinished
        }
    }

    /// This sorting algorithm should have a runtime of O(n) if only one new element is inserted.
    /// After a shrinkage, the runtime is O(n log n).
    #[inline]
    fn order_simplex(&self, simplex: &mut Simplex) {
        simplex.sort_unstable_by(|&(fa, _), &(fb, _)| fa.approx_cmp_ulps(&fb, self.ulps));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis};

    #[test]
    fn test_far() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        let n = 4;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();
        // let mut x0 = Array1::zeros(n);
        // x0[1] = 1.0;
        let x: Array2<f64> = (Array::eye(n) + Array1::ones(n)) / (n + 1) as f64;
        let (exit_status, xout) = nm.minimize(&f, x);
        eprintln!("{:?}", exit_status);
        eprintln!("{:?}", xout);
        assert!(f(xout.view()).abs() <  1e-7);
        let correct = Array1::ones(n) / n as f64;
        assert!(correct.all_close(&xout, 1e-5));
    }

    #[test]
    fn test_f_close() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        let n = 5;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();        

        let eps = 1f64 / n as f64;
        let x0 = Array1::ones(n) / n as f64;
        let x: Array2<f64> = Array::eye(x0.len()) * eps + &x0 * (1.0-eps);

        let (exit_status, xout) = nm.minimize(&f, x);

        eprintln!("{:?}", exit_status);
        eprintln!("{:?}", xout);
        let correct = Array1::ones(n) / n as f64;
        assert!(f(xout.view()).abs() <  1e-7);
        assert!(correct.all_close(&xout, 1e-5));
    }
}