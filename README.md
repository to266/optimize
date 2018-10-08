# optimize

This crate is a fork of to266/optimize (optimize on crates.io). 
This version is very unstable and poorly documented to favor rapid development.

 * The project has been restructured into modules for scalar and vector functions
 * The `Minimize` Trait has been dropped to emphasize the differences between algorithms
 * Nelder-Mead has seen some optimizations, requiring less memory allocations and less numerical operations per iteration 