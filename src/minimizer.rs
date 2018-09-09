
use ndarray::prelude::*;

/// A general minimizer trait.
pub trait Minimizer {
    /// Minimizes the given function returned scalar value by exploring the parameter space.
    /// May or may not use numerical differential, depending on particular implementation.
    fn minimize<F: Fn(ArrayView1<f64>) -> f64>(
        &self,
        func: F,
        args: ArrayView1<f64>,
    ) -> Array1<f64>;
}
