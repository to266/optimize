//! This implementation of Nelder-Mead is based on
//!  
//! Gao, F and Han, L. Implementing the Nelder-Mead simplex algorithm with 
//! adaptive parameters. 2012. Computational Optimization and Applications. 
//! 51:1, pp 259--277
//! 
//! In particular, it adapts their suggestion to use adaptive step sizes,
//! which depend on the dimensionality of the optimization problem.
//! 
//! # Use case
//! 
//! The Nelder-Mead algorithm does not require a gradient or a hessian.
//! As a tradeoff it typically requires a lot of function evaluations to 
//! find a minimum. Further, there are few theoretical results on the 
//! convergence of Nelder-Mead iterations.
//! 
//! # Examples
//! 
//! ```
//! # extern crate ndarray;
//! # extern crate optimize;
//! # use ndarray::prelude::*;
//! # use optimize::vector::NelderMeadBuilder;
//! 
//! let function =
//!     |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
//! let minimizer = NelderMeadBuilder::default()
//!     .xtol(1e-8)
//!     .ftol(1e-8)
//!     .build()
//!     .unwrap();
//! let args = Array::from_vec(vec![3.0, -8.3]);
//! let res = minimizer.minimize(&function, args.view());
//! println!("res: {}", res);
//! ```

use float_cmp::ApproxOrdUlps;
use ndarray::prelude::*;
use ::utils::WrappedFunction;

type Simplex = Vec<(f64, Array1<f64>)>;


#[derive(Builder, Debug)]
/// A minimizer for a scalar function of one or more variables using the Nelder-Mead algorithm.
pub struct NelderMead {
    /// The required number of floating point representations that separate two numbers to consider them
    /// equal. See crate float_cmp for more information.
    #[builder(default = "1")]
    pub ulps: i64,

    /// The maximum number of iterations to optimize. If neither maxiter nor maxfun are given, both
    /// default to n*200 where n is the number of parameters to optimize.
    #[builder(default = "None")]
    #[builder(setter(into))]
    pub maxiter: Option<usize>,

    /// The maximum number of function calls used to optimize. If neither maxiter nor maxfun are given, both
    /// default to n*200 where n is the number of parameters to optimize.
    #[builder(default = "None")]
    #[builder(setter(into))]
    pub maxfun: Option<usize>,

    /// Adapt algorithm parameters to dimensionality of the problem. Useful for high-dimensional minimization.
    #[builder(default = "false")]
    pub adaptive: bool,

    /// Absolute error in function parameters between iterations that is acceptable for convergence.
    #[builder(default = "1e-4f64")]
    pub xtol: f64,

    /// Absolute error in function values between iterations that is acceptable for convergence.
    #[builder(default = "1e-4f64")]
    pub ftol: f64,
}

impl NelderMead {

    /// Search for the value minimizing `func` given an initial guess
    /// in the form of a point. The algorithm will explore the variable
    /// space without constraints.
    pub fn minimize<F>(&self, func: F, x0: ArrayView1<f64>) -> Array1<f64>
    where F: Fn(ArrayView1<f64>) -> f64 {
        let n = x0.len();
        let eps = 0.05;
        let mut init_simplex = Array2::default((n+1, n));
        init_simplex.slice_mut(s![0, ..]).assign(&x0);
        init_simplex.slice_mut(s![1.., ..]).assign(&(Array::eye(n) * eps + &x0 * (1.0-eps)));

        self.minimize_simplex(func, init_simplex)
    }

    /// Search for the value minimizing `func` given an initial guess
    /// in the form of a set of coordinates, the `init_simplex`. This algorithm
    /// only ever explores the space spanned by these initial vectors.
    /// If you have parameter restrictions that effectively place your parameters 
    /// in a subspace, you can enforce these restrictions by setting `init_simplex`
    /// to a basis of this subspace.
    pub fn minimize_simplex<F>(&self, func: F, init_simplex: Array2<f64>) -> Array1<f64>
    where F: Fn(ArrayView1<f64>) -> f64 {
        let mut func = WrappedFunction { num: 0, func: func };
        let mut simplex = init_simplex.outer_iter().map(|xi| (func.call(xi), xi.to_owned())).collect::<Simplex>();
        self.order_simplex(&mut simplex);
        let mut centroid = self.centroid(&simplex);
        let n = simplex.len();

        let (maxiter, maxfun, alpha, beta, gamma, delta) = self.initialize_parameters(n);

        let mut iterations = 1;

        while !self.finished(&simplex, iterations, maxiter, func.num, maxfun) {

            let f_n1 = simplex[n-1].0;
            let f_n = simplex[n-2].0;
            let f_0 = simplex[0].0;

            let reflected = &centroid + &(alpha * &(&centroid - &simplex[n-1].1));
            let f_reflected = func.call(reflected.view());

            if f_reflected < f_n && f_reflected > f_0 { // try reflecting the worst point through the centroid
                self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
            } else if f_reflected < f_0 { // try expanding beyond the centroid
                let expanded = &centroid + &(beta * &(&reflected - &centroid));
                let f_expanded = func.call(expanded.view());

                if f_expanded < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, expanded, f_expanded);
                } else {
                    self.lean_update(&mut simplex, &mut centroid, reflected, f_reflected);
                }
            } else if f_reflected < f_n1 && f_reflected >= f_n { // try a contraction outwards
                let contracted = &centroid + &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted = func.call(contracted.view());
                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // shrink
                    self.shrink(&mut simplex, &mut func, delta, &mut centroid);
                }
            } else { // try a contraction inwards
                let contracted = &centroid - &(gamma * (&centroid - &simplex[n-1].1));
                let f_contracted = func.call(contracted.view());

                if f_contracted < f_reflected {
                    self.lean_update(&mut simplex, &mut centroid, contracted, f_contracted);
                } else {
                    // shrink
                    self.shrink(&mut simplex, &mut func, delta, &mut centroid);
                }
            }
            iterations += 1;
        }
        simplex.remove(0).1
    }

    /// Helper function to keep the main loop clean. Resolves default values that can
    /// only be known after the minimize function is called.
    #[inline]
    fn initialize_parameters(&self, n: usize) -> (usize, usize, f64, f64, f64, f64) {
        let maxiter = match self.maxiter {
            Some(x) => x,
            None => 200 * n
        };
        let maxfun = match self.maxfun {
            Some(x) => x,
            None => 200 * n
        };

        let (alpha, beta, gamma, delta) = if self.adaptive {
            let dim = n as f64;
            (1.0, 1.0 + 2.0 / dim, 0.75 - 1.0 / (2.0 * dim), 1.0 - 1.0 / dim)
        } else {
            (1.0, 2.0, 0.5, 0.5)
        };

        (maxiter, maxfun, alpha, beta, gamma, delta)
    }

    #[inline]
    fn finished(&self, simplex: &Simplex, iterations: usize, maxiter: usize, nfeval: usize, maxfun: usize) -> bool {
        let n = simplex.len();
        iterations > maxiter 
        || nfeval > maxfun 
        || ( simplex[n-1].0 - simplex[0].0 < self.ftol 
             && (&simplex[n-1].1 - &simplex[0].1).mapv(f64::abs).scalar_sum() < n as f64 * self.xtol )
    }

    /// Update the centroid effiently, knowing only one value changed.
    /// The pattern-defeating sort of order_simplex is allready efficient
    /// given that we inserted a single out-of-place value in a sorted vec.
    /// This update is O(n).
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
    /// The centroid is updated by shrinking the centroid directly,
    /// Then removing the new 'worst x' and adding in the old 'worst x'.
    /// This update of `centroid` is O(n). 
    /// Shrinkage requires n function evaluations.
    #[inline]
    fn shrink<F>(&self, simplex: &mut Simplex, f: &mut WrappedFunction<F>, sigma: f64, centroid: &mut Array1<f64>)
    where F: Fn(ArrayView1<f64>) -> f64 {
        {
            let mut iter = simplex.iter_mut();
            let (_, x0) = iter.next().unwrap();
            for (fi, xi) in iter {
                *xi *= sigma;
                *xi += &((1.0 - sigma) * &x0.view());
                *fi = f.call(xi.view());
            }
        }
        let n = simplex.len() - 1;
        let old_worst = simplex[n - 1].1.to_owned();
        *centroid *= sigma;
        *centroid += &((1.0 - sigma) * &simplex[0].1);
        self.order_simplex(simplex);
        *centroid += &((&old_worst - &simplex[n - 1].1) / (n - 1) as f64);
    }

    /// calculate the centroid of all points but the worst one.
    /// Assumes that the simplex is ordered. This calculation is O(n^2).
    #[inline]
    fn centroid(&self, simplex: &Simplex) -> Array1<f64> {
        let n = simplex.len();
        let mut centroid = Array1::zeros(simplex[0].1.len());
        for (_, xi) in simplex.iter().take(n-1) {
            centroid += xi;
        }
        centroid / (n-1) as f64
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
    use float_cmp::ApproxEq;

    #[test]
    fn simplex() {
        let function =
            |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let minimizer = NelderMeadBuilder::default()
            .xtol(1e-8)
            .ftol(1e-8)
            .build()
            .unwrap();
        let args = Array::from_vec(vec![3.0, -8.3]);
        let res = minimizer.minimize(&function, args.view());
        println!("res: {}", res);
        assert!(res[0].approx_eq(&1.0, 1e-4, 10));
        assert!(res[1].approx_eq(&1.0, 1e-4, 10));
    }

}
