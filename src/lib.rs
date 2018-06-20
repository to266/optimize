extern crate ndarray;
extern crate num_traits;

// use ndarray::prelude::*;
use std::fmt::Debug;

// use num_traits::NumOps;

pub struct Minimizer {}

impl Minimizer {
    pub fn minimize<F, A, R>(&self, func: F, args: &A) 
    where
        F: Fn(&A) -> R,
        A: ?Sized,
        R: Debug
    {
        let ans = func(args);
        println!("calculated this: {:?}", ans);
    }

    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
