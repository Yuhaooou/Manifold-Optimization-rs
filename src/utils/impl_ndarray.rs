use ndarray::{ScalarOperand, prelude::*};

use crate::utils::traits::{InnerProduct, Norm as VectorNorm, RCLike, Vector};

impl<D, IxN> Vector for Array<D, IxN>
where
    D: RCLike + ScalarOperand,
    IxN: Dimension,
{
    type Field = D;

    fn sum(&self) -> Self::Field {
        ArrayRef::<D, IxN>::sum(self)
    }

    fn ref_add_num(&self, num: Self::Field) -> Self {
        self + num
    }

    fn ref_sub_num(&self, num: Self::Field) -> Self {
        self - num
    }

    fn ref_mul_num(&self, num: Self::Field) -> Self {
        self * num
    }

    fn ref_div_num(&self, num: Self::Field) -> Self {
        self / num
    }

    fn ref_add(&self, rhs: Self) -> Self {
        self + rhs
    }

    fn ref_sub(&self, rhs: Self) -> Self {
        self - rhs
    }

    fn ref_add_ref(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn ref_sub_ref(&self, rhs: &Self) -> Self {
        self - rhs
    }

    fn elementwise_mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn elementwise_div(&self, rhs: &Self) -> Self {
        self / rhs
    }

    fn zeros_like(&self) -> Self {
        Array::<D, IxN>::zeros(self.raw_dim())
    }

    fn ones_like(&self) -> Self {
        Array::<D, IxN>::from_elem(self.raw_dim(), D::one())
    }

    fn nums_like(&self, elem: Self::Field) -> Self {
        Array::<D, IxN>::from_elem(self.raw_dim(), elem)
    }
}

impl<K, IxN> VectorNorm for Array<K, IxN>
where
    K: RCLike,
    IxN: Dimension,
{
    type Field = K;

    fn norm(&self) -> K {
        self.iter().map(|x| x.powi(2)).sum::<K>().sqrt()
    }
}

impl<K, IxN> InnerProduct for Array<K, IxN>
where
    K: RCLike,
    IxN: Dimension,
{
    type Field = K;

    // Need optimization.
    fn inner(&self, rhs: &Array<K, IxN>) -> K {
        assert!(
            self.shape() == rhs.shape(),
            "Inner product requires the same shape"
        );
        (self * rhs.mapv(K::conjugate)).sum()
    }
}
