use ndarray::prelude::*;

pub struct WrappedFunction<F: Fn(ArrayView1<f64>) -> f64> {
    pub num: usize,
    pub func: F,
}

impl<F: Fn(ArrayView1<f64>) -> f64> WrappedFunction<F> {
    pub fn call(&mut self, arg: ArrayView1<f64>) -> f64 {
        self.num += 1;
        (self.func)(arg)
    }
}

pub fn approx_fprime<F>(xk: ArrayView1<f64>, func: F, epsilon: ArrayView1<f64>) -> Array1<f64>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    let f0 = func(xk);
    let n = xk.len();
    let mut grad = Array1::<f64>::zeros(n);
    let mut ei = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    for k in 0..n {
        ei[k] = 1.0;
        d.assign(&(&ei * &epsilon));
        grad[k] = (func((&d + &xk).view()) - &f0) / &d[k];
        ei[k] = 0.0;
    }
    grad
}


#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::ApproxEq;

    #[test]
    fn gradient() {
        let function = |x: ArrayView1<f64>| 1.0 * x[0].powi(2) + 200. * x[1].powi(2);
        let x = Array::from_vec(vec![1.0, 1.0]);
        let eps_ar = Array::from_vec(vec![1e-7, 14.14 * 1e-7]);
        let res = approx_fprime(x.view(), function, eps_ar.view());

        println!("Res: {}", res);
        assert!(res[0].approx_eq(&2.0, 1e-4, 10));
        assert!(res[1].approx_eq(&400.0, 1e-3, 10));
    }

}
