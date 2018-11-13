/// Limited-memory BFGS Quasi-Newton optimizer. Uses the two-loop recursion to
/// calculate the quasi-inverse-hessian, as formulated in
///
/// Jorge Nocedal. Updating Quasi-Newton Matrices With Limited Storage.
/// MATHEMATICS OF  COMPUTATION, VOLUME 35,  NUMBER 151 JULY 1980, PAGES 773-782
///
use ndarray::prelude::*;
use std::collections::VecDeque;

#[derive(Builder, Debug)]
pub struct LBFGS {
    /// Smaller is more precise.
    #[builder(default = "1e-12")]
    pub gtol: f64,

    /// The maximum number of iterations. The gradient is evaluated once per iteration.
    /// Larger is more precise.
    #[builder(default = "1500")]
    pub max_iter: usize,

    /// The number of datapoints to use to estimate the inverse hessian.
    /// Larger is more precise. If `m` is larger than `x0.len()`, then
    /// `x0.len()` is used.
    #[builder(default = "5")]
    pub m: usize,

    /// The maximum step to be taken in the direction determined by the Quasi-Newton
    /// method.
    #[builder(default = "2.0")]
    pub max_step: f64,

    /// The tolerance on x used to terminate the line search.
    /// Smaller is more precise.
    #[builder(default = "1e-8")]
    pub xtol: f64,

    /// The maximum number of function evaluations.
    #[builder(default = "1500")]
    pub max_feval: usize,
}

impl LBFGS {
    /// Minimize `func` starting in `x0` using a finite difference approximation
    /// for the gradient. For more control over the approximation of the gradient,
    /// you can use `minimize` with you own approximation.
    pub fn minimize_approx_grad<F>(&self, func: F, x0: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let eps = Array1::ones(x0.len()) * 1e-9;
        let eps_view = eps.view();
        let grad = |x: ArrayView1<f64>| ::utils::approx_fprime(x, &func, eps_view);
        self.minimize(&func, &grad, x0)
    }

    /// Minimize `func` starting in `x0` using `grad` as the gradient of `func`.
    pub fn minimize<F, G>(&self, func: F, grad: G, x0: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
        G: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        let mut iter = 0;
        let mut feval_count = 0;
        let m = x0.len().min(self.m).max(1);

        let mut hist = VecDeque::with_capacity(m);

        let mut x = x0.to_owned();
        let mut g = grad(x.view());

        loop {
            let dir = self.quasi_update(&g, &hist);
            let a = {
                let min = ::scalar::GoldenRatioBuilder::default()
                    .xtol(self.xtol)
                    .build()
                    .unwrap();
                let f = |a: f64| {
                    feval_count += 1;
                    func((&x + &(a * &dir)).view())
                };
                min.minimize_bracket(f, -self.max_step, 0.0)
            };
            let x_new = &x + &(a * &dir);
            let g_new = grad(x_new.view());

            let s = &x_new - &x;
            let y = &g_new - &g;
            let r = 1f64 / s.dot(&y);

            iter += 1;

            if r.is_nan()
                || iter > self.max_iter
                || feval_count > self.max_feval
                || g_new.mapv(f64::abs).scalar_sum() < self.gtol
            {
                break;
            }

            while hist.len() >= m {
                hist.pop_front();
            }
            hist.push_back((s, y, r));

            x = x_new;
            g = g_new;
        }

        x
    }

    /// Calculate the Quasi-Newton direction H*g efficiently, where g
    /// is the gradient and H is the *inverse* hessian.
    fn quasi_update(
        &self,
        grad: &Array1<f64>,
        hist: &VecDeque<(Array1<f64>, Array1<f64>, f64)>,
    ) -> Array1<f64> {
        let mut q = grad.to_owned();
        let mut a = Vec::with_capacity(hist.len());

        for (si, yi, ri) in hist.iter().rev() {
            let ai = ri * si.dot(&q);
            q.scaled_add(-ai, &yi);
            a.push(ai);
        }

        // q = {
        //     // H_0 * q
        //     let (ref s, ref y, _) = hist[hist.len() - 1];
        //     y * (s.dot(&q) / y.dot(y))
        // };

        for ((si, yi, ri), ai) in hist.iter().zip(a.iter().rev()) {
            let bi = ri * yi.dot(&q);
            q.scaled_add(ai - bi, &si);
        }
        q
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn minimize() {
        let center = arr1(&[0.9, 1.3, 0.5]);
        let min = LBFGSBuilder::default().build().unwrap();
        let f = |x: ArrayView1<f64>| (&x - &center).mapv(|xi| -(-xi * xi).exp()).scalar_sum();
        let g = |x: ArrayView1<f64>| {
            -2.0 * (&x - &center).mapv(|xi| -(-xi * xi).exp()) * &(&x - &center)
        };
        let x0 = Array1::ones(center.len());
        let xmin = min.minimize(&f, &g, x0.view());
        println!("{:?}", xmin);
        assert!(xmin.all_close(&center, 1e-5))
    }

    #[test]
    fn minimize_approx() {
        let center = arr1(&[0.9, 1.3, 0.5]);
        let min = LBFGSBuilder::default().build().unwrap();
        let f = |x: ArrayView1<f64>| (&x - &center).mapv(|xi| -(-xi * xi).exp()).scalar_sum();
        let x0 = Array1::ones(center.len());
        let xmin = min.minimize_approx_grad(&f, x0.view());
        println!("{:?}", xmin);
        assert!(xmin.all_close(&center, 1e-5))
    }
}
