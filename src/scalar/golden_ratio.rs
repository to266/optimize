//! Golden ratio is to minimization what bisection is to root finding.
//! GoldenRatio searches within an interval for a local minimum. At
//! every iteration, the interval is decreased in size by a constant
//! factor, until the desired precision is obtained.
//!
//! This algorithm is guaranteed to converge on a local minimum in a
//! finite amount of steps under very light smoothess criteria for the
//! target function.
//!
//! In an iteration, the target function is calculated at 4 points:
//!         +---------+----+---------+
//! iter 1  a         b    c         d
//! The interval for the next iteration is chosen to be [a,c] if b<c,
//! and [b,d] if c<=b. The distances b-a, c-b, and d-c are chosen in such
//! a way that 3 out of 4 points can be reused, and only 1 new function
//! evaluation is required in the next iteration. If b<c this looks like:
//!         +---------+----+---------+
//! iter 1  a         b    c         d
//!         +----+----+----+
//! iter 2  a    b    c    d

use std::f64;

#[derive(Builder, Debug)]
pub struct GoldenRatio {
    /// The width of the interval at which convergence is satisfactory.
    /// Smaller is more precise.
    #[builder(default = "1e-8")]
    pub xtol: f64,

    /// The maximum number of iterations before the search terminates.
    /// The number of function evaluations in a bracket search is
    /// 2 + <number of iterations>.
    /// Bigger is more precise.
    #[builder(default = "1000")]
    pub max_iter: usize,
}

/// The golden ratio is 1.0 : 0.5 + .5*5f64.sqrt(). Therefore we need the ratio
/// 1.0 / (1.5 + .5*5f64.sqrt()) to calculate the fractions in which to divide
/// intervals.
const RATIO: f64 = 2.618033988749895; // 1.5 + 0.5*5f64.sqrt();

impl GoldenRatio {
    /// Search for the minimum of `func` around `x0`.
    /// The search region is expanded in both directions until an expansion
    /// leads to an increase in the function value at the bound of the region.
    ///
    /// Currently `minimize` makes, in some specific cases, at most  around 2000
    /// additional calls to `func` to find a bracketing interval. For more details
    /// see https://github.com/to266/optimize/issues/11
    pub fn minimize<F>(&self, mut func: F, x0: f64) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let left = x0 + self.explore(&mut func, x0, true);
        let right = x0 + self.explore(&mut func, x0, false);
        self.minimize_bracket(func, left, right)
    }

    /// increase step until func stops decreasing, then return step
    fn explore<F>(&self, mut func: F, x0: f64, explore_left: bool) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let mut step = if explore_left { -1. } else { 1. }
            * f64::powi(2., f64::log2(f64::EPSILON + x0.abs()) as i32)
            * f64::EPSILON;

        let mut fprev = func(x0);
        let mut fstepped = func(x0 + step);

        while fstepped < fprev {
            step *= 2.0;
            fprev = fstepped;
            fstepped = func(x0 + step);
        }
        step
    }

    /// Search for the minimum of `func` between `left` and `right`.
    pub fn minimize_bracket<F>(&self, mut func: F, left: f64, right: f64) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let mut min = left.max(f64::MIN);
        let mut max = right.min(f64::MAX);
        let mut iter = 0;

        let mut x_b = min + (max - min) / RATIO;
        let mut x_c = max - (max - min) / RATIO;
        let mut f_b = func(x_b);
        let mut f_c = func(x_c);

        while (max - min).abs() > self.xtol && iter < self.max_iter {
            iter += 1;
            if f_b < f_c {
                max = x_c;
                x_c = x_b;
                x_b = min + (max - min) / RATIO;

                f_c = f_b;
                f_b = func(x_b);
            } else {
                min = x_b;
                x_b = x_c;
                x_c = max - (max - min) / RATIO;

                f_b = f_c;
                f_c = func(x_c);
            }
        }
        (min + max) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio_bracketed() {
        let minimizer = GoldenRatioBuilder::default()
            .xtol(1e-7)
            .max_iter(1000)
            .build()
            .unwrap();
        let f = |x: f64| (x - 0.2).powi(2);
        let res = minimizer.minimize_bracket(&f, -1.0, 1.0);

        println!("res: {}", res);
        assert!((res - 0.2).abs() <= 1e-7);
    }

    #[test]
    fn golden_ratio_x0() {
        let minimizer = GoldenRatioBuilder::default()
            .xtol(1e-7)
            .max_iter(1000)
            .build()
            .unwrap();
        let f = |x: f64| (x - 0.2).powi(2);
        let res = minimizer.minimize(&f, 10.0);

        println!("res: {}", res);
        assert!((res - 0.2).abs() <= 1e-7);
    }

    #[test]
    fn monotone_x0() {
        let minimizer = GoldenRatioBuilder::default()
            .xtol(1e-7)
            .max_iter(1000)
            .build()
            .unwrap();
        let f = |x: f64| x;
        let res = minimizer.minimize(&f, 10.0);

        println!("res: {}", res);
        assert!(res.is_infinite());
    }
}
