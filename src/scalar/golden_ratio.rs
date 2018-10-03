use ::Status;

/// Golden ratio is to minimization what bisection is to root finding.
pub struct GoldenRatio {
    pub xtol: f64,
    pub ftol: f64,
}

const RATIO: f64 = 2.61803398875;//1.5 + 0.5*f64::sqrt(5.0);

impl GoldenRatio {
    pub fn new() -> Self {
        GoldenRatio {
            xtol: 1e-4,
            ftol: 1e-4
        }
    }
    
    pub fn minimize(&self, func: &Fn(f64) -> f64, left: f64, right: f64) -> (f64, Status) {
        let mut status = Status::NotFinished;
        let mut min = left;
        let mut max = right;

        let mut x_b = min + (max - min) / RATIO;
        let mut x_c = max - (max - min) / RATIO;
        let mut f_b = func(x_b);
        let mut f_c = func(x_c);

        while status == Status::NotFinished {
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
            status = self.update_status(min, max, f_b, f_c);
        }
        ((min + max) / 2.0, status)
    }

    #[inline]
    fn update_status(&self, min: f64, max: f64, f_b: f64, f_c: f64) -> Status {
        if (f_b - f_c).abs() < self.ftol {
            return Status::FtolConvergence
        } else if (max-min).abs() < self.xtol {
            return Status::XtolConvergence
        } else {
            return Status::NotFinished
        }
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio() {
        let mut minimizer = GoldenRatio::new();
        minimizer.xtol = 1e-7;
        minimizer.ftol = 1e-6;
        let f = |x: f64| (x-0.2).powi(2);
        let (res, status) = minimizer.minimize(&f, -1.0, 1.0);

        println!("res: {}", res);
        println!("status: {:?}", status);
        assert!( (res - 0.2).abs() <= 1e-7 || (f(res) -0.0).abs() <= 1e-6 );
    }
}