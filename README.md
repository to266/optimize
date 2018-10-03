# optimize

This crate started as a fork of to266/optimize (optimize on crates.io) but I decided to rewrite it in its entirity. 
Different optimization methods serve different purposes, and therefore I do not think a minimizer trait is appropriate.
Every method can have its own signature, stressing the relevance of things like bounds or initial points.

Usage:

```
extern crate optimize;
use optimize::scalar::GoldenRatio;

let minimizer = GoldenRation::new().ftol(1e-5).xtol(1e-5);
let min = minimizer.minimize(|x: f64| x*x, -0.5, 0.5);
println!("minimum found at {}", min);
```