# optimize

[![Crates.io](https://img.shields.io/crates/v/optimize.svg)](https://crates.io/crates/optimize)
[![Documentation](https://docs.rs/optimize/badge.svg)](https://docs.rs/optimize/)
[![Build Status](https://travis-ci.org/to266/optimize.svg?branch=master)](https://travis-ci.org/to266/optimize)

This crate provides (non-linear) numerical optimization methods.

It is heavily based on `scipy.optimize`.

The crate is actively developed and expanded to include more methods.

A simple example follows:

```rust
// Define a function that we aim to minimize
let function = |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);

// Create a minimizer using the builder pattern. If some of the parameters are not given, default values are used.
let minimizer = NelderMeadBuilder::default()
    .xtol(1e-6f64)
    .ftol(1e-6f64)
    .maxiter(50000)
    .build()
    .unwrap();

// Set the starting guess
let args = Array::from_vec(vec![3.0, -8.3]);

// Run the optimization
let ans = minimizer.minimize(&function, args.view());

// Print the optimized values
println!("Final optimized arguments: {:?}", ans);
```
