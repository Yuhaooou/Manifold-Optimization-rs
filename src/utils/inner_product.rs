use ndarray::prelude::*;

use crate::utils::traits::RCLike;

/// Inner product extension trait for ndarray arrays.
pub trait InnerProduct {
    type Elem: RCLike;
    type IxN: Dimension;

    /// Return the Euclidean inner product with `rhs`.
    fn inner(&self, rhs: &Array<Self::Elem, Self::IxN>) -> Self::Elem;
}

impl<A, IxN> InnerProduct for Array<A, IxN>
where
    A: RCLike,
    IxN: Dimension,
{
    type Elem = A;
    type IxN = IxN;

    // Need optimization.
    fn inner(&self, rhs: &Array<A, IxN>) -> A {
        assert!(
            self.shape() == rhs.shape(),
            "Inner product requires the same shape"
        );
        (self * rhs.mapv(RCLike::conj)).sum()
    }
}
