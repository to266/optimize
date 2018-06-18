extern crate ndarray;
extern crate num_traits;

use ndarray::prelude::*;
use std::fmt::Debug;

use num_traits::NumOps;

pub struct Minimizer {
}

impl<'a> Minimizer {

    pub fn minimize<F, A:'a, R: 'a, N: 'a>(&self, func: F, args: A)
    where F: Fn(A) -> &'a R,
          A: Clone + AsArray<'a, N>,
          R: AsArray<'a, N> + Debug,
          N: NumOps
    {
        let mut ans = func(args);
        println!("{:?}", ans);
    }

    pub fn new() -> Self {
        Self{}
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
