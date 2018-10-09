use float_cmp::{ApproxEqUlps, ApproxOrdUlps};
use minimizer;
use ndarray::prelude::*;
use ndarray::Zip;
use utils;

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
    fn order_simplex(&self, mut sim: ArrayViewMut2<f64>, mut fsim: ArrayViewMut1<f64>) {
        // order sim[0,..] by fsim
        let mut _tmp = unsafe { Array2::<f64>::uninitialized(sim.dim()) };
        let mut order: Vec<usize> = (0..fsim.len()).collect();
        order.sort_unstable_by(|&a, &b| fsim[a].approx_cmp_ulps(&fsim[b], self.ulps));
        for (k, s) in order.iter().enumerate() {
            _tmp.slice_mut(s![k, ..]).assign(&sim.slice(s![*s, ..]));
        }
        sim.assign(&_tmp);
        // order fsim
        let mut _tmp = fsim.to_vec();
        _tmp.sort_unstable_by(|&a, b| a.approx_cmp_ulps(b, self.ulps));
        fsim.assign(&Array1::<f64>::from_vec(_tmp));
    }
}

impl minimizer::Minimizer for NelderMead {
    fn minimize<F>(&self, func: F, args: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let mut func = utils::WrappedFunction { num: 0, func: func };
        let mut x0 = Array1::<f64>::from_iter(args.iter().map(|&v| v as f64));
        let n: usize = x0.len();

        let adaptive = self.adaptive;

        let maxiter: Option<usize>;
        let maxfun: Option<usize>;
        match (self.maxiter, self.maxfun) {
            (None, None) => {
                maxiter = Some(200 * n);
                maxfun = Some(200 * n);
            }
            _ => {
                maxiter = self.maxiter;
                maxfun = self.maxfun;
            }
        }

        let rho: f64;
        let chi: f64;
        let psi: f64;
        let sigma: f64;

        if adaptive {
            let dim = n as f64;
            rho = 1f64;
            chi = 1f64 + 2. / dim;
            psi = 0.75 - 1. / (2. * dim);
            sigma = 1. - 1. / dim;
        } else {
            rho = 1f64;
            chi = 2f64;
            psi = 0.5f64;
            sigma = 0.5f64;
        }

        let nonzdelt = 0.05f64;
        let zdelt = 0.00025f64;

        let mut sim = Array2::<f64>::zeros((n + 1, n));
        sim.slice_mut(s![0, ..]).assign(&x0);
        for k in 0..n {
            let mut y = x0.clone();
            y[k] = match y[k].approx_eq_ulps(&0., self.ulps) {
                true => zdelt,
                _ => (1. + nonzdelt) * y[k],
            };
            sim.slice_mut(s![k + 1, ..]).assign(&y);
        }
        let mut fsim = Array1::<f64>::zeros(n + 1);

        Zip::from(&mut fsim).and(sim.genrows()).apply(|f, s| {
            *f = func.call(s);
        });

        self.order_simplex(sim.view_mut(), fsim.view_mut());
        let mut iterations: usize = 1;
        let s0 = s![0, ..];
        let sm1 = s![-1, ..];

        loop {
            if maxiter.map_or(false, |v| v <= iterations) || maxfun.map_or(false, |v| v <= func.num)
            {
                break;
            }
            if (&sim.slice(s![1.., ..]) - &sim.slice(s0))
                .mapv(f64::abs)
                .fold(0., |acc, &x| match acc.approx_gt_ulps(&x, self.ulps) {
                    true => acc,
                    false => x,
                })
                .approx_lt_ulps(&self.xtol, self.ulps)
            {
                break;
            }
            if (&fsim.slice(s![1..]) - fsim[0])
                .mapv(f64::abs)
                .fold(0., |acc, &x| match acc.approx_gt_ulps(&x, self.ulps) {
                    true => acc,
                    false => x,
                })
                .approx_lt_ulps(&self.ftol, self.ulps)
            {
                break;
            }

            let xbar = sim
                .slice(s![..-1, ..])
                .genrows()
                .into_iter()
                .fold(Array1::<f64>::zeros(n), |acc, x| acc + x) / n as f64;

            let xr = &xbar * (rho + 1.) - (&sim.slice(sm1) * rho);
            let fxr = func.call(xr.view());

            let mut doshrink = false;

            if fxr.approx_lt_ulps(&fsim[0], self.ulps) {
                let xe = (1. + rho * chi) * &xbar - (rho * chi * &sim.slice(sm1));
                let fxe = func.call(xe.view());

                if fxe.approx_lt_ulps(&fxr, self.ulps) {
                    sim.slice_mut(sm1).assign(&xe);
                    fsim[n] = fxe;
                } else {
                    sim.slice_mut(sm1).assign(&xr);
                    fsim[n] = fxr;
                }
            } else {
                if fxr.approx_lt_ulps(&fsim[n - 1], self.ulps) {
                    sim.slice_mut(sm1).assign(&xr);
                    fsim[n] = fxr;
                } else {
                    if fxr.approx_lt_ulps(&fsim[n], self.ulps) {
                        let xc = (1. + psi * rho) * &xbar - psi * rho * &sim.slice(sm1);
                        let fxc = func.call(xc.view());

                        if fxc.approx_gt_ulps(&fxr, self.ulps) {
                            doshrink = true;
                        } else {
                            sim.slice_mut(sm1).assign(&xc);
                            fsim[n] = fxc;
                        }
                    } else {
                        let xcc = (1. - psi) * &xbar + psi * &sim.slice(sm1);
                        let fxcc = func.call(xcc.view());

                        if fxcc.approx_lt_ulps(&fsim[n], self.ulps) {
                            sim.slice_mut(sm1).assign(&xcc);
                            fsim[n] = fxcc;
                        } else {
                            doshrink = true;
                        }
                    }

                    if doshrink {
                        for j in 1..n + 1 {
                            let sj = s![j, ..];
                            let mut _tmp = &sim.slice(sj) - &sim.slice(s0);
                            _tmp *= sigma;
                            _tmp += &sim.slice(s0);
                            sim.slice_mut(sj).assign(&_tmp);
                            fsim[j] = func.call(sim.slice(sj).view());
                        }
                    }
                }
            }

            self.order_simplex(sim.view_mut(), fsim.view_mut());
            iterations += 1;
        }
        x0.assign(&sim.slice(s![0, ..]));
        x0
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::ApproxEq;
    use minimizer::Minimizer;

    #[test]
    fn simplex() {
        let function =
            |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let minimizer = NelderMeadBuilder::default()
            .xtol(1e-8f64)
            .ftol(1e-8f64)
            .build()
            .unwrap();
        let args = Array::from_vec(vec![3.0, -8.3]);
        let res = minimizer.minimize(&function, args.view());
        println!("res: {}", res);
        assert!(res[0].approx_eq(&1.0, 1e-4, 10));
        assert!(res[1].approx_eq(&1.0, 1e-4, 10));
    }
}
