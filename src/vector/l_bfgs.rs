/// Limited-memory BFGS Quasi-Newton optimizer. Uses the two-loop recursion to
/// calculate the quasi-inverse-hessian, as formulated in
///
/// Jorge Nocedal. Updating Quasi-Newton Matrices With Limited Storage.
/// MATHEMATICS OF  COMPUTATION, VOLUME 35,  NUMBER 151 JULY 1980, PAGES 773-782
///
use ndarray::prelude::*;

#[derive(Builder, Debug)]
pub struct LBFGS {
    /// Smaller is more precise.
    #[builder(default = "1e-8")]
    pub gtol: f64,

    /// Larger is more precise.
    #[builder(default = "1500")]
    pub max_iter: usize,
}

impl LBFGS {
    pub fn minimize<F, G>(&self, func: F, grad: G, x0: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
        G: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        let mut iter = 0;
        let m = x0.len().min(5);

        let mut hist = RobinVec::new();

        let mut x = x0.to_owned();
        let mut g = grad(x.view());

        loop {
            let dir = self.quasi_update(&g, &hist);
            let a = {
                let min = ::scalar::GoldenRatioBuilder::default().build().unwrap();
                let f = |a: f64| func((&x + &(a * &dir)).view());
                min.minimize_bracket(&f, -2.0, 0.0)
            };
            let x_new = &x + &(a * &dir);
            let g_new = grad(x_new.view());

            let s = &x_new - &x;
            let y = &g_new - &g;
            let r = 1f64 / s.dot(&y);

            iter += 1;

            if r.is_nan() || iter > self.max_iter || g_new.mapv(f64::abs).scalar_sum() < self.gtol {
                break;
            }

            hist.push((s, y, r), m);

            x = x_new;
            g = g_new;
        }

        x
    }

    fn quasi_update(
        &self,
        grad: &Array1<f64>,
        hist: &RobinVec<(Array1<f64>, Array1<f64>, f64)>,
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

#[derive(Debug)]
struct RobinVec<T> {
    i0: usize,
    vec: Vec<T>,
}

use std::iter::Chain;
use std::slice;

impl<T> RobinVec<T> {
    pub fn new() -> RobinVec<T> {
        RobinVec {
            i0: 0,
            vec: Vec::new(),
        }
    }

    pub fn iter(&self) -> Chain<slice::Iter<T>, slice::Iter<T>> {
        self.vec[self.i0..].iter().chain(self.vec[..self.i0].iter())
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn push(&mut self, el: T, size: usize) {
        let n = self.vec.len();
        if size > n {
            if self.i0 == 0 {
                self.vec.push(el);
            } else {
                self.vec.insert(self.i0, el);
                self.i0 += 1;
            }
        } else if size == n {
            self.vec[self.i0] = el;
            self.i0 = (self.i0 + 1) % n;
        } else {
            panic!("needs implementation")
        }
    }
}

use std::ops::{Index, IndexMut};
impl<T> Index<usize> for RobinVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.vec[(index + self.i0) % self.vec.len()]
    }
}

impl<T> IndexMut<usize> for RobinVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let n = self.vec.len();
        &mut self.vec[(index + self.i0) % n]
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
    fn robin() {
        let mut r = RobinVec::new();
        for i in 1..16 {
            r.push(i, 4);
        }
        println!("{:?}", r);
        for (i, &ri) in r.iter().enumerate() {
            println!("{}", ri);
            assert_eq!(i + 12, ri);
            assert_eq!(ri, r[i]);
        }
    }
}
