//! This module provides the base framework for all minimizers present in this crate, such as the
//! base trait and return type.
use ndarray::prelude::*;
use std::time::Duration;

/// Minimizer states at the end of the run
#[derive(Debug, PartialEq)]
pub enum RunStatus {
    /// Minimizer finished successfully.
    Success,
    /// Minimization was not succesful
    Failure,
}

/// Final minimization parameters that result in the smallest found value.
#[derive(Debug, PartialEq)]
pub enum MinResult {
    /// Scalar parameter value where the minimum is found
    Scalar(f64),
    /// Vector of parameters where the minimum if found
    Vector(Array1<f64>),
}

/// A minimization result, storing various details of the run and the final results.
#[derive(Debug, PartialEq)]
pub struct OptimResult {
    /// The runtime of the minimization according to the system clock.
    pub runtime: Option<Duration>,
    /// The number of function evaluations performed.
    pub f_evals: Option<usize>,
    /// The number of iterations run.
    pub iterations: Option<usize>,
    /// The final parameter values.
    pub minimum: Option<MinResult>,
    /// The function value at the found minimum.
    pub minimum_value: Option<f64>,
    /// The minimizer success or failure status.
    pub status: RunStatus,
}

/// A general minimizer trait.
pub trait Minimizer {
    /// Minimizes the given function returned scalar value by exploring the parameter space.
    /// May or may not use numerical differential, depending on particular implementation.
    fn minimize<F: Fn(ArrayView1<f64>) -> f64>(
        &self,
        func: F,
        args: ArrayView1<f64>,
    ) -> OptimResult;
}
