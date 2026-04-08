use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::{Lapack, SVD, Scalar};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::distr::uniform::SampleUniform;

use crate::manifolds::Manifold;
use crate::manifolds::manifold::{EGradToRGrad, EHessToRHess, RandomPoint};
use crate::utils::traits::{RCLike, Real};
use crate::utils::{
    inner_product::InnerProduct,
    tools::{get_scalar_from_float, mat_sym, qr},
};

#[derive(Debug, Clone)]
/// Stiefel manifold `St(n, p)` of orthonormal `n x p` matrices.
pub struct Stiefel<D>
where
    D: RCLike,
{
    name: String,
    n: usize,
    p: usize,
    retraction_type: u8,
    _marker: std::marker::PhantomData<D>,
}

impl<D> Stiefel<D>
where
    D: RCLike,
{
    /// Create `St(n, p)` with default QR retraction.
    pub fn new(n: usize, p: usize) -> Self {
        assert!(n >= p && p >= 1, "Need n >= p >= 1");
        Stiefel {
            name: format!("Stiefel manifold St({},{})", n, p),
            n,
            p,
            retraction_type: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set retraction type: `0` for QR, `1` for Polar.
    pub fn set_retraction_type(&mut self, retraction_type: u8) -> &mut Self {
        if retraction_type > 1 {
            panic!("Invalid retraction type. Must be 0 (QR) or 1 (Polar).");
        }
        self.retraction_type = retraction_type;
        self
    }

    fn retraction_qr(point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D>
    where
        D: Scalar + ScalarOperand + Lapack + Real,
    {
        qr(&(point + tangent_vector)).unwrap()
    }

    fn retraction_polar(point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D>
    where
        D: Scalar + ScalarOperand + Lapack + Real,
    {
        let (u, _, vt) = (point + tangent_vector).svd(true, true).unwrap();
        u.unwrap().dot(&vt.unwrap())
    }
}

impl<D> Manifold for Stiefel<D>
where
    D: Real + ScalarOperand + Scalar + Lapack,
{
    type Point = Array2<D>;
    type TangentVector = Array2<D>;
    type AmbientPoint = Array2<D>;
    type Field = D;

    fn base_point(&self) -> Self::Point {
        let mut res = Array2::zeros((self.n, self.p));
        for i in 0..self.p {
            res[(i, i)] = D::one();
        }
        res
    }

    fn zero_tangent_vector(&self, _point: &Self::Point) -> Self::TangentVector {
        Array2::zeros((self.n, self.p))
    }

    #[inline]
    fn inner(
        &self,
        _point: &Self::Point,
        tangent_vector1: &Array2<D>,
        tangent_vector2: &Array2<D>,
    ) -> D {
        tangent_vector1.inner(tangent_vector2)
    }

    fn projection(&self, point: &Array2<D>, ambient_point: &Self::TangentVector) -> Self::Point {
        let tmp1 = (Array::eye(self.n) - point.dot(&point.t())).dot(ambient_point);
        let tmp2 = point.t().dot(ambient_point) - ambient_point.t().dot(point);
        tmp1 + point.dot(&tmp2) / get_scalar_from_float::<D>(2.)
    }

    fn retraction(&self, point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D> {
        match self.retraction_type {
            0 => Self::retraction_qr(point, tangent_vector),
            1 => Self::retraction_polar(point, tangent_vector),
            _ => unimplemented!("Invalid retraction type"),
        }
    }
}

impl<D> EGradToRGrad for Stiefel<D>
where
    D: Real + ScalarOperand + Scalar + Lapack,
{
    fn egrad_to_rgrad(&self, point: &Array2<D>, egrad: &Array2<D>) -> Array2<D> {
        egrad - point.dot(&mat_sym(&point.t().dot(egrad)))
    }
}

impl<D> EHessToRHess for Stiefel<D>
where
    D: Real + ScalarOperand + Scalar + Lapack,
{
    fn ehess_to_rhess(
        &self,
        point: &Array2<D>,
        tangent_vector: &Array2<D>,
        egrad: &Array2<D>,
        ehess: &Array2<D>,
    ) -> Array2<D> {
        let tmp = ehess - tangent_vector.dot(&mat_sym(&point.t().dot(egrad)));
        self.projection(point, &tmp)
    }
}

impl<D> RandomPoint for Stiefel<D>
where
    D: Real + Scalar + ScalarOperand + SampleUniform + Lapack,
{
    /// Sample a random matrix and retract to the manifold.
    fn random_point(&self) -> Array2<D> {
        let distribution = Uniform::new(
            get_scalar_from_float::<D>(0.),
            get_scalar_from_float::<D>(1.),
        )
        .unwrap();
        let point = Array2::random((self.n, self.p), distribution);
        self.retraction(&self.base_point(), &point)
    }
}

#[allow(unused_imports)]
mod tests {
    use super::*;
    use ndarray_linalg::Norm;

    #[test]
    fn test_stiefel() {
        let n = 5;
        let p = 3;
        let eps = 1e-10;
        let manifold = Stiefel::<f64>::new(n, p);

        let point = manifold.random_point();
        let err1 = (point.t().dot(&point) - Array2::<f64>::eye(p)).norm_l2();
        assert!(
            err1 < eps,
            "Point is not on the Stiefel manifold: error = {}",
            err1
        );

        let ambient_point = Array2::random((n, p), Uniform::new(0., 1.).unwrap());
        let tangent_vector = manifold.projection(&point, &ambient_point);
        let xtv = point.t().dot(&tangent_vector);
        let err2 = (&xtv + &xtv.t()).norm_l2();
        assert!(
            err2 < eps,
            "Tangent vector is not in the tangent space: error = {}",
            err2
        );

        let retracted_point = manifold.retraction(&point, &(tangent_vector * 1.2));
        let err3 = (retracted_point.t().dot(&retracted_point) - Array2::<f64>::eye(p)).norm_l2();
        assert!(
            err3 < eps,
            "Retracted point is not on the Stiefel manifold: error = {}",
            err3
        );
    }
}
