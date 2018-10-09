mod nelder_mead;
pub use self::nelder_mead::NelderMead;

mod line_search;
pub use self::line_search::LineSearch;

use ndarray::{ArrayView1, Array1};

pub fn approximate_gradient_neg<'a>(f: &Target, x: ArrayView1<f64>) -> Array1<f64> {
    let eps = 1e-8;
    let f0 = f(x);
    let mut dir = x.to_owned();
    let mut grad = x.to_owned();
    for i in 0..x.len() {
        dir[i] += eps;
        grad[i] = ( f(dir.view()) - f0 ) / eps;
        dir[i] -= eps;
    }
    -grad
}

/// bounded_step checks if the simplex does not walk outside of the convex search region. 
/// To enforce convex constraints, implement your own version of bounded_step where you 
/// limit the stepsize in <direction>.
fn unbounded_step(max_growth: f64, _direction: ArrayView1<f64>, _from: ArrayView1<f64>) -> f64 {
    max_growth
}


pub type Target<'a> = Fn(ArrayView1<f64>)->f64 + 'a;
pub type Bound<'a> = Fn(f64, ArrayView1<f64>, ArrayView1<f64>) -> f64 + 'a;
pub type Gradient<'a> = Fn(&Target, ArrayView1<f64>) -> Array1<f64> + 'a;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_gradient() {
        let n = 5;
        let f = |x: ArrayView1<f64>| (&x - 0.5).mapv(|xi| xi*xi).scalar_sum();
        let x0 = Array1::ones(n)*1.5;
        let g = approximate_gradient_neg(&f, x0.view());        
        println!("{:?}", g);
        assert!(g.all_close(&(&Array1::ones(n)*-2.0), 1e-7));
    }
}