use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::types::Scalar;
use ndarray_linalg::{Lapack, Norm};

use crate::utils::traits::{InnerProduct, Norm as VectorNorm, RCLike, Real, Vector};

impl<D, IxN> Vector for Array<D, IxN>
where
    D: RCLike + Clone + ScalarOperand + Scalar,
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

impl<D, IxN> VectorNorm for Array<D, IxN>
where
    D: Real + Lapack<Real = D>,
    IxN: Dimension,
{
    type Real = D;

    fn norm(&self) -> D {
        self.norm_l2()
    }
}

impl<D, IxN> InnerProduct for Array<D, IxN>
where
    D: Real,
    IxN: Dimension,
{
    type Real = D;

    fn inner(&self, rhs: &Array<D, IxN>) -> D {
        (self * rhs).sum()
    }
}
