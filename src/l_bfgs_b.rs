use super::Minimizer;


#[derive(Builder, Debug)]
/// A minimizer for a scalar function of one or more variables using the Nelder-Mead algorithm.
pub struct L_BFGS_B {
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


impl Minimizer for NelderMead {
    fn minimize<F>(&self, func: F, args: ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        // TODO
    }
}