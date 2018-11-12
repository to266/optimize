//! This module contains algorithms that search for local minima of functions along multiple dimensions.

mod nelder_mead;

pub use self::nelder_mead::NelderMeadBuilder;

mod l_bfgs;
pub use self::l_bfgs::LBFGSBuilder;
