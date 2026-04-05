use ndarray::prelude::*;
use ndarray_linalg::Scalar;

use crate::utils::traits::Real;

/// Inner product extension trait for ndarray arrays.
pub trait InnerProduct {
    type Elem: Scalar;
    type IxN: Dimension;

    /// Return the Euclidean inner product with `rhs`.
    fn inner(&self, rhs: &Array<Self::Elem, Self::IxN>) -> Self::Elem;
}

impl<A, IxN> InnerProduct for Array<A, IxN>
where
    A: Real + Scalar,
    IxN: Dimension,
{
    type Elem = A;
    type IxN = IxN;

    fn inner(&self, rhs: &Array<A, IxN>) -> A {
        assert!(
            self.shape() == rhs.shape(),
            "Inner product requires the same shape"
        );
        (self * rhs).sum()
    }
}
