use ndarray::{ArrayView1, ArrayViewMut1, Array, Array1, Array2, Axis};
use float_cmp::{ApproxOrdUlps};
use ::Status;

pub struct NelderMead {
    pub ulps: i64,
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
}

type Simplex = Vec<(f64, Array1<f64>)>;
type Target = Fn(ArrayView1<f64>)->f64;

impl NelderMead {
    
    pub fn new() -> Self {
        NelderMead {
            ulps: 1,
            max_iter: 1000,
            ftol: 1e-9,
            xtol: 1e-9,
        }
    }

    pub fn minimize(&mut self, f: &Target, mut x0: ArrayViewMut1<f64>) -> Status {
        if (x0.scalar_sum() - 1.0).abs() > x0.len() as f64 * ::std::f64::EPSILON {
            panic!("The initial point does not lie on the simplex.");
        }
        let mut status = Status::NotFinished;
        let alpha = 1.0; // > 0; reflection multiplier
        let gamma = 2.0; // > 1; expansion multiplier
        let rho = 0.5;  // > 0, < 0.5; contraction factor
        let sigma = 0.5; // > 0, < 1; shrinkage factor
        let n = x0.len();
        let eps = 1.0;//1.0 / (n*n) as f64;

        // initialize array of start points
        let mut simplex; {
            let x: Array2<f64> = Array::eye(x0.len()) * eps + &x0 * (1.0-eps);
            simplex = x.outer_iter().map(|xi| (f(xi), xi.to_owned())).collect::<Simplex>();
            self.order_simplex(&mut simplex);
        }

        let mut iter = 0;

        while status == Status::NotFinished {

            let centroid = self.centroid(&simplex);
            let f_worst = simplex[n-1].0;
            let f_best = simplex[0].0;

            // attempt reflecting f_worst through the centroid
            let reflected = self.bounded_step(alpha, (&centroid - &simplex[n-1].1).view(), centroid.view());
            let f_reflected = f(reflected.view());

            if f_reflected < f_worst && f_reflected > f_best {
                simplex[n-1] = (f_reflected, reflected);
            } else if f_reflected < f_best { // try expanding beyond the centroid
                let expanded = self.bounded_step(gamma, (&centroid - &reflected).view(), centroid.view());
                let f_expanded = f(expanded.view());

                simplex[n-1] =  if f_expanded < f_reflected {(f_expanded, expanded)} else {(f_reflected, reflected)};
            } else { // try a contraction towards the centroid
                let contracted = &centroid - &(rho * (&centroid - &simplex[n-1].1));
                let f_contracted = f(contracted.view());

                if f_contracted < f_worst {
                    simplex[n-1] = (f_contracted, contracted);
                } else { // shrink if all else fails
                    self.shrink(&mut simplex, sigma, f);
                }
            }        
            self.order_simplex(&mut simplex);
            iter += 1;
            eprint!("\r{}: {}", iter, f_best);
            status = self.update_status(iter, &simplex);
        }
        x0.assign(&simplex[0].1);
        status
    }

    #[inline]
    fn shrink(&self, simplex: &mut Simplex, sigma: f64, f:&Target) {
        let mut iter = simplex.iter_mut();
        let (_, x0) = iter.next().unwrap();
        for (fi, xi) in iter {
            *xi *= sigma;
            *xi += &((1.0-sigma) * &x0.view());
            *fi = f(xi.view());
        }
    }

    #[inline]
    /// calculate the centroid of all points but the worst one.
    fn centroid(&self, simplex: &Simplex) -> Array1<f64> {
        let n = simplex.len();
        let mut centroid = Array1::zeros(simplex[0].1.len());
        for (_, xi) in simplex.iter().take(n-1) {
            centroid += xi;
        }
        centroid / (n-1) as f64
    }

    /// Calculate the max step size we can take without leaving the simplex.
    /// To remedy round-off errors, take the elementwise max with 0.0.
    /// TODO generalize this a bit. Allow the caller to supply a simple convex bound-check function
    #[inline]
    fn bounded_step(&self, max_growth: f64, direction: ArrayView1<f64>, from: ArrayView1<f64>) -> Array1<f64> {
        let delta = direction.iter().zip(&from)
            .map(|(d,f)| if *d<0.0 {f/d.abs()} else {max_growth})
            .fold(max_growth, |acc, x| if acc < x {acc} else {x});

        let vec = (&from + &(delta * &direction)).mapv(|xi| 0f64.max(xi));
        let s = vec.scalar_sum();
        vec / s
    }

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

    #[inline]
    fn order_simplex(&self, simplex: &mut Simplex) {
        simplex.sort_unstable_by(|&(fa, _), &(fb, _)| fa.approx_cmp_ulps(&fb, self.ulps));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use rand::thread_rng;
    use rand::distributions::{Distribution, Normal};

    #[test]
    fn test_far() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 2000;
        let n = 4;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();
        let mut x0 = Array1::zeros(n);
        x0[1] = 1.0;
        let exit_status = nm.minimize(&f, x0.view_mut());
        eprintln!("{:?}", exit_status);
        eprintln!("{:?}", x0);
        let correct = Array1::ones(n) / n as f64;
        assert!(correct.all_close(&x0, 1e-5));
    }

    #[test]
    fn test_close() {
        let mut nm = NelderMead::new();
        nm.ftol = 1e-9;
        nm.max_iter = 5000;
        let n = 5;
        let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();        
        let mut x0 = Array1::ones(n) / n as f64;
        let exit_status = nm.minimize(&f, x0.view_mut());
        eprintln!("{:?}", exit_status);
        eprintln!("{:?}", x0);
        let correct = Array1::ones(n) / n as f64;
        assert!(correct.all_close(&x0, 1e-5));
    }

    #[test]
    fn test_rebalance() {
        let normal = Normal::new(0.0, 0.01);
        let mut rng = thread_rng();
        let t = 250;
        let k = 500;
        let iter = normal.sample_iter(&mut rng);
        let mut x = Array1::ones(k) / k as f64;
        let mut r = Array1::from_iter(iter.take(t*k)).into_shape((k,t)).unwrap();
        r.mapv_inplace(f64::exp);
        
        let f = move |x: ArrayView1<f64>| {
            let x_ = x.into_shape((x.len(), 1)).unwrap();
            let ultimo = &r * &x_;
            let growth = ultimo.sum_axis(Axis(0));
            let diff = &ultimo/&growth - x_;
            let tx_cost = diff.mapv(f64::abs).sum_axis(Axis(0)) * 0.002;
            let log_wealth = (growth - tx_cost).mapv(f64::ln).scalar_sum();
            -log_wealth
        };

        let mut nm = NelderMead::new();
        nm.max_iter = 2000;
        println!("{:?}", nm.minimize(&f, x.view_mut()));
    }

    #[bench]
    fn bench_close(bench: &mut Bencher) {
        bench.iter(|| {
            let mut nm = NelderMead::new();
            nm.ftol = 1e-9;
            nm.max_iter = 5000;
            let n = 50;
            let f = |x: ArrayView1<f64>| (&x - &x.mean_axis(Axis(0))).mapv(f64::abs).scalar_sum();        
            let mut x0 = Array1::ones(n) / n as f64;
            print!("{:?}", nm.minimize(&f, x0.view_mut()));
        });
    }

    #[bench]
    fn bench_rebalance(bench: &mut Bencher) {
        bench.iter(|| test_rebalance());
    }
}