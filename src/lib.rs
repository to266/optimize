#[macro_use(s)]
extern crate ndarray;

extern crate float_cmp;
extern crate num_traits;

use float_cmp::{ApproxEqUlps, ApproxOrdUlps};
use ndarray::prelude::*;
use ndarray::Zip;

static ULPS: i64 = 1;

pub struct Minimizer {}

fn order_simplex(mut sim: ArrayViewMut2<f64>, mut fsim: ArrayViewMut1<f64>) {
    // order sim[0,..] by fsim
    let mut _tmp = unsafe { Array2::<f64>::uninitialized(sim.dim()) };
    let mut order: Vec<usize> = (0..fsim.len()).collect();
    order.sort_unstable_by(|&a, &b| fsim[a].approx_cmp_ulps(&fsim[b], ULPS));
    for (k, s) in order.iter().enumerate() {
        _tmp.slice_mut(s![k, ..]).assign(&sim.slice(s![*s, ..]));
    }
    sim.assign(&_tmp);
    // order fsim
    let mut _tmp = fsim.to_vec();
    _tmp.sort_unstable_by(|&a, b| a.approx_cmp_ulps(b, ULPS));
    fsim.assign(&Array1::<f64>::from_vec(_tmp));
}

impl Minimizer {
    pub fn minimize<F>(&self, func: F, args: ArrayView1<f64>)
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let ans = func(args);
        println!("calculated this: {:?}", ans);
        self.minimize_neldermead(func, args);
    }

    fn minimize_neldermead<F>(&self, func: F, args: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let mut x0 = args.to_owned();
        let n: usize = x0.len();

        let rho = 1f64;
        let chi = 2f64;
        let psi = 0.5f64;
        let sigma = 0.5f64;
        let xatol = 1e-4f64;

        let mut sim = Array2::<f64>::zeros((n + 1, n));
        let nonzdelt = 0.05f64;
        let zdelt = 0.00025f64;
        sim.slice_mut(s![0, ..]).assign(&x0);
        for k in 0..n {
            let mut y = x0.clone();
            y[k] = match y[k].approx_eq_ulps(&0., ULPS) {
                true => zdelt,
                _ => (1. + nonzdelt) * y[k],
            };
            sim.slice_mut(s![k + 1, ..]).assign(&y);
        }
        let maxiter = 200 * n;
        let mut fsim = Array1::<f64>::zeros(n + 1);

        Zip::from(&mut fsim).and(sim.genrows()).apply(|f, s| {
            *f = func(s);
        });

        order_simplex(sim.view_mut(), fsim.view_mut());
        let mut iterations: usize = 1;
        let s0 = s![0, ..];
        let sm1 = s![-1, ..];

        while iterations < maxiter {
            if (&sim.slice(s![1.., ..]) - &sim.slice(s0))
                .mapv(f64::abs)
                .fold(0., |acc, &x| match acc.approx_gt_ulps(&x, ULPS) {
                    true => acc,
                    false => x,
                })
                .approx_lt_ulps(&xatol, ULPS)
            {
                break;
            }

            let xbar = sim.slice(s![..-1, ..])
                .genrows()
                .into_iter()
                .fold(Array1::<f64>::zeros(n), |acc, x| acc + x) / n as f64;

            let xr = &xbar * (rho + 1.) - (&sim.slice(sm1) * rho);
            let fxr = func(xr.view());

            let mut doshrink = false;

            if fxr.approx_lt_ulps(&fsim[0], ULPS) {
                let xe = (1. + rho * chi) * &xbar - (rho * chi * &sim.slice(sm1));
                let fxe = func(xe.view());

                if fxe.approx_lt_ulps(&fxr, ULPS) {
                    sim.slice_mut(sm1).assign(&xe);
                    fsim[n] = fxe;
                } else {
                    sim.slice_mut(sm1).assign(&xr);
                    fsim[n] = fxr;
                }
            } else {
                if fxr.approx_lt_ulps(&fsim[n - 1], ULPS) {
                    sim.slice_mut(sm1).assign(&xr);
                    fsim[n] = fxr;
                } else {
                    if fxr.approx_lt_ulps(&fsim[n], ULPS) {
                        let xc = (1. + psi * rho) * &xbar - psi * rho * &sim.slice(sm1);
                        let fxc = func(xc.view());

                        if fxc.approx_gt_ulps(&fxr, ULPS) {
                            doshrink = true;
                        } else {
                            sim.slice_mut(sm1).assign(&xc);
                            fsim[n] = fxc;
                        }
                    } else {
                        let xcc = (1. - psi) * &xbar + psi * &sim.slice(sm1);
                        let fxcc = func(xcc.view());

                        if fxcc.approx_lt_ulps(&fsim[n], ULPS) {
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
                            fsim[j] = func(sim.slice(sj).view());
                        }
                    }
                }
            }

            order_simplex(sim.view_mut(), fsim.view_mut());
            iterations += 1;
        }
        x0.assign(&sim.slice(s![0, ..]));
        println!("Final value: {}\nArgs:\n{}", fsim[0], x0);
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
