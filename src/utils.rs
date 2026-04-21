//! Shared numeric helpers and abstractions used across the crate.

use ndarray::Array2;

use crate::utils::{
    lapack::LapackElem, qr::LinalgQR, svd::LinalgSVD, traits::RCLike
};

pub mod impl_ndarray;
pub mod lapack;
pub mod qr;
pub mod svd;
pub mod tools;
pub mod traits;

mod private {
    use crate::utils::lapack::LapackElem;

    pub trait Sealed: LapackElem {}

    impl<T> Sealed for T where T: LapackElem {}
}

pub trait LinalgBase {
    type Elem: private::Sealed;
}

impl<T> LinalgBase for Array2<T>
where
    T: LapackElem,
{
    type Elem = T;
}

pub trait Linalg: LinalgBase + LinalgQR + LinalgSVD {}

impl<T> Linalg for Array2<T>
where
    T: RCLike,
    Array2<T>: LinalgBase + LinalgQR + LinalgSVD,
{
}

