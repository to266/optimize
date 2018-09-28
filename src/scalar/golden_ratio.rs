#[derive(Builder)]
pub struct GoldenRatio {
    /// Absolute error in function parameters between iterations that is acceptable for convergence.
    #[builder(default = "1e-4f64")]
    pub xtol: f64,

    /// Absolute error in function values between iterations that is acceptable for convergence.
    #[builder(default = "1e-4f64")]
    pub ftol: f64,
}

const RATIO: f64 = 2.61803398875;//1.5 + 0.5*f64::sqrt(5.0);

impl super::super::BoundedScalarMinimizer for GoldenRatio {

    fn minimize<F>(&self, func: F, x0: f64, left: f64, right: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut min = left;
        let mut max = right;

        let mut x_b = min + (max - min) / RATIO;
        let mut x_c = max - (max - min) / RATIO;
        let mut f_b = func(x_b);
        let mut f_c = func(x_c);

        while (max-min).abs() > self.xtol || (f_b - f_c).abs() > self.ftol {
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
    use super::super::super::BoundedScalarMinimizer;

    #[test]
    fn golden_ratio() {
        let minimizer = GoldenRatioBuilder::default()
            .xtol(1e-7)
            .ftol(1e-6)
            .build()
            .unwrap();
        let f = |x: f64| (x-0.2).powi(2);
        let res = minimizer.minimize(&f, 0.0, -1.0, 1.0);

        println!("res: {}", res);
        assert!( (res - 0.2).abs() <= 1e-7 );
        assert!( (f(res) -0.0).abs() <= 1e-6 );
    }
}