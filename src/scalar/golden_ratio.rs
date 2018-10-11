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

#[derive(Builder, Debug)]
pub struct GoldenRatio {
    /// The width of the interval at which convergence is satisfactory. 
    /// Smaller is more precise.
    #[builder(default = "1e-8")]
    pub xtol: f64,

    /// The maximum number of iterations before the search terminates.
    /// Bigger is more precise.
    #[builder(default = "1000")]
    pub max_iter: usize,
}

const RATIO: f64 = 2.61803398875;//1.5 + 0.5*f64::sqrt(5.0);

impl GoldenRatio {
    
    /// The main minimization routine. Searches for the minimum of `func`
    /// between `left` and `right`.
    pub fn minimize<F>(&self, func: F, left: f64, right: f64) -> f64 
    where F: Fn(f64) -> f64 {
        let mut min = left;
        let mut max = right;
        let mut iter = 0;

        let mut x_b = min + (max - min) / RATIO;
        let mut x_c = max - (max - min) / RATIO;
        let mut f_b = func(x_b);
        let mut f_c = func(x_c);

        while (max-min).abs() > self.xtol && iter < self.max_iter {
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
    fn golden_ratio() {
        let minimizer = GoldenRatioBuilder::default()
            .xtol(1e-7)
            .max_iter(1000)
            .build().unwrap();
        let f = |x: f64| (x-0.2).powi(2);
        let res = minimizer.minimize(&f, -1.0, 1.0);

        println!("res: {}", res);
        assert!( (res - 0.2).abs() <= 1e-7);
    }
}