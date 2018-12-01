//! Algorithms that search for local minima of functions along multiple dimensions.

mod nelder_mead;

pub use self::nelder_mead::NelderMead;
pub use self::nelder_mead::NelderMeadBuilder;

mod l_bfgs;
pub use self::l_bfgs::LBFGSBuilder;
pub use self::l_bfgs::LBFGS;
