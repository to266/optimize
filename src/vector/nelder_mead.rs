use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Array, Array1, Array2, Axis};
use float_cmp::{ApproxOrdUlps};

pub struct NelderMead {
    ulps: i64,
    max_iter: usize,
    ftol: f64,
    xtol: f64,
    iter: usize
}

#[derive(Debug)]
enum Action {
    Contract,
    Expand,
    Reflect,
    Shrink
}

impl NelderMead {
    pub fn new() -> Self {
        NelderMead {
            ulps: 1,
            max_iter: 1000,
            ftol: 1e-9,
            xtol: 1e-9,
            iter: 0
        }
    }

    pub fn max_iter(&mut self, i: usize) {self.max_iter = i;}
    pub fn ftol(&mut self, ftol: f64) {self.ftol = ftol;}
    pub fn xtol(&mut self, xtol: f64) {self.xtol = xtol;}

    pub fn minimize(&mut self, f: &Fn(ArrayView1<f64>)->f64, mut x0: ArrayViewMut1<f64>) {
        if (x0.scalar_sum() - 1.0).abs() > x0.len() as f64 * ::std::f64::EPSILON {
            panic!("The initial point does not lie on the simplex.");
        }

        let alpha = 1.0; // > 0
        let gamma = 2.0; // > 1
        let rho = 0.5;  // > 0, < 0.5
        let sigma = 0.5; // > 0, < 1
        let n = x0.len();
        let mut eps = 1.0 / n as f64;

        // initialize array of start points
        let mut x: Array2<f64> = Array::eye(x0.len()) * eps + &x0 * (1.0-eps);

        // initialize function values
        let mut fx: Array1<f64> = Array1::from_shape_fn(x0.len(), |i| f(x.row(i)));

        // if the simplex is too small to start, increase its size
        // while eps < 0.5 && !self.not_finished(fx.view(), x.view()) {
        //     eps *= 2.0;
            
        //     x = Array::eye(x0.len()) * eps + &x0 * (1.0-eps);
        //     fx = Array1::from_shape_fn(x0.len(), |i| f(x.row(i)));
        // }

        self.order_simplex(x.view_mut(), fx.view_mut());
        let mut tmp: Array1<f64> = Array1::zeros(n);
        self.iter = 0;

        while self.not_finished(fx.view(), x.view()){
            self.iter += 1;
            eprint!("\r{}: {}", self.iter, fx[0]);

            let centroid = x.slice(s![..-1, ..]).mean_axis(Axis(0));
            let f_worst = fx[n-1];

            // attempt reflection
            let reflected = self.bounded_step(alpha, (&centroid - &x.row(n-1)).view(), centroid.view());
            assert_simplex(reflected.view(), Action::Reflect);
            let f_reflected = f(reflected.view());

            if f_reflected < f_worst && f_reflected > fx[0] {
                // reflection successful, re-sort x and fx
                self.reorder_simplex(x.view_mut(), fx.view_mut(), tmp.view_mut(), f_reflected, reflected);              

            // else attempt expansion
            } else if f_reflected < fx[0] {

                let expanded = self.bounded_step(gamma, (&centroid - &reflected).view(), centroid.view());
                let f_expanded = f(expanded.view());
                assert_simplex(expanded.view(), Action::Expand);

                if f_expanded < f_reflected {
                    self.reorder_simplex(x.view_mut(), fx.view_mut(), tmp.view_mut(), f_expanded, expanded);               
                } else {
                    self.reorder_simplex(x.view_mut(), fx.view_mut(), tmp.view_mut(), f_reflected, reflected);             
                }

            // else try a contraction
            } else {
                let contracted = &centroid - &(rho * (&centroid - &x.row(n-1))); // needs no checking because it is a convex combination
                let f_contracted = f(contracted.view());
                assert_simplex(contracted.view(), Action::Contract);

                if f_contracted < f_worst {
                    self.reorder_simplex(x.view_mut(), fx.view_mut(), tmp.view_mut(), f_contracted, contracted);                

                // else shrink
                } else {
                    {
                        let mut iter = x.outer_iter_mut();
                        tmp.assign(&iter.next().unwrap());
                        for mut xi in iter {
                            xi *= sigma;
                            xi += &((1.0-sigma) * &tmp);
                        }
                    }
                    fx = Array1::from_shape_fn(x0.len(), |i| f(x.row(i))); 
                    self.order_simplex(x.view_mut(), fx.view_mut());
                }
            }        
        }
        x0.assign(&x.slice(s![0, ..]));
    }

    /// Calculate the max step size we can take without leaving the simplex.
    /// To remedy round-off errors, take the elementwise max with 0.0.
    fn bounded_step(&self, max_growth: f64, direction: ArrayView1<f64>, from: ArrayView1<f64>) -> Array1<f64> {
        let delta = direction.iter().zip(&from)
            .map(|(d,f)| if *d<0.0 {f/d.abs()} else {max_growth})
            .fold(max_growth, |acc, x| if acc < x {acc} else {x});

        (&from + &(delta * &direction)).mapv(|xi| 0f64.max(xi))
    }

    /// Terminate the algorithm if 
    /// the best and worst f differ by only a small amount,
    /// and the best and worst x differ only by a small amount,
    /// and we haven't crossed the max_iter limit.
    fn not_finished(&self, f: ArrayView1<f64>, x: ArrayView2<f64>) -> bool {
        let n = f.len();
        let ftol = f[n-1] - f[0];
        let xtol = (&x.slice(s![n-1, ..]) - &x.slice(s![0, ..])).mapv(f64::abs).scalar_sum();
        self.iter < self.max_iter 
        && (ftol > self.ftol || xtol > self.xtol)
    }

    fn reorder_simplex(&self, mut x: ArrayViewMut2<f64>, mut fx: ArrayViewMut1<f64>, mut tmp: ArrayViewMut1<f64>, fnew: f64, xnew: Array1<f64>) {
        for i in (0..xnew.len()).rev() {
            if i == 0 || fx[i-1] < fnew {
                fx[i] = fnew;
                x.slice_mut(s![i, ..]).assign(&xnew);
                break;
            } else {
                fx[i] = fx[i-1];
                tmp.slice_mut(s![..]).assign(&x.slice(s![i-1, ..]));
                x.slice_mut(s![i, ..]).assign(&tmp);
            }
        }
    }

    fn order_simplex(&self, mut sim: ArrayViewMut2<f64>, mut fsim: ArrayViewMut1<f64>) {
        // order sim[0,..] by fsim
        let mut _tmp = Array2::<f64>::zeros(sim.dim());
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

fn assert_simplex(x: ArrayView1<f64>, action: Action) {
    if (x.scalar_sum() - 1.0) > 1e-12
        || !x.fold(true, |acc, xi| acc && *xi >= 0.0) {
            let min = x.fold(1.0, |acc,xi| if acc<*xi{acc}else{*xi});
            let max = x.fold(0.0, |acc,xi| if acc>*xi{acc}else{*xi});
            panic!("vector outside of simplex produced by {:?}.\nasserts: {},{}\nsum: {}\nxmin:{}, xmax:{}",
             action, (x.scalar_sum() - 1.0) > 1e-12, x.fold(true, |acc, xi| acc && *xi >= 0.0), x.scalar_sum(), min, max);
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use test::Bencher;

    #[test]
    fn test_far() {
        let mut nm = NelderMead::new();
        nm.ftol(1e-9);
        nm.max_iter(2000);
        let n = 4;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();
        let mut x0 = Array1::zeros(n);
        x0[1] = 1.0;
        nm.minimize(&f, x0.view_mut());
        eprintln!("{:?}", x0);
        let correct = Array1::ones(n) / n as f64;
        assert!(correct.all_close(&x0, 1e-5));
    }

    #[test]
    fn test_close() {
        let mut nm = NelderMead::new();
        nm.ftol(1e-9);
        nm.max_iter(5000);
        let n = 200;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();        
        let mut x0 = Array1::ones(n) / n as f64;
        nm.minimize(&f, x0.view_mut());
        eprintln!("{:?}", x0);
        let correct = Array1::ones(n) / n as f64;
        assert!(correct.all_close(&x0, 1e-5));
    }

    // #[bench]
    // fn bench_close(bench: &mut Bencher) {
    //     bench.iter(|| test_close());
    // }
}