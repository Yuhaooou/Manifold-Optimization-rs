use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::Lapack;
use ndarray_rand::RandomExt;
use num_complex::ComplexFloat;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::manifolds::{EGradToRGrad, EHessToRHess, Exp, Manifold, RandomPoint};
use crate::random_point_forward;
use crate::utils::svd::{SVDBackend, thin_svd_owned, thin_svd_ref};
use crate::utils::traits::InnerProduct;
use crate::utils::traits::RCLike;

#[derive(Debug, Clone)]
/// Grassmann manifold `Gr(n, p)` of `p`-dimensional subspaces in `R^n`.
pub struct Grassmann<D>
where
    D: RCLike,
{
    pub name: String,
    n: usize,
    p: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D> Grassmann<D>
where
    D: RCLike,
{
    /// Create `Gr(n, p)` with `n >= p >= 1`.
    pub fn new(n: usize, p: usize) -> Self {
        assert!(n >= p && p >= 1, "Need n >= p >= 1");
        Grassmann {
            name: format!("Grassmann manifold Gr({},{})", n, p),
            n,
            p,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

impl<D> Manifold for Grassmann<D>
where
    D: RCLike + ScalarOperand + Lapack,
{
    type Point = Array2<D>;
    type TangentVector = Array2<D>;
    type AmbientPoint = Array2<D>;
    type Field = D;

    fn base_point(&self) -> Self::Point {
        todo!()
    }

    fn zero_tangent_vector(&self, _point: &Self::Point) -> Self::TangentVector {
        todo!()
    }

    fn inner(
        &self,
        _point: &Self::Point,
        tangent_vector1: &Self::TangentVector,
        tangent_vector2: &Self::TangentVector,
    ) -> Self::Field {
        tangent_vector1.inner(tangent_vector2)
    }

    fn projection(&self, point: &Self::Point, ambient: &Self::AmbientPoint) -> Self::TangentVector {
        ambient - point.dot(&point.t().dot(ambient))
    }

    fn retraction(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point {
        let (u, _, vt) = thin_svd_owned(point + tangent_vector, SVDBackend::GESDD, None);
        u.dot(&vt)
    }
}

impl<D> EGradToRGrad for Grassmann<D>
where
    D: RCLike + ScalarOperand + Lapack,
{
    fn egrad_to_rgrad(
        &self,
        point: &Self::Point,
        egrad: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        self.projection(point, egrad)
    }
}

impl<D> EHessToRHess for Grassmann<D>
where
    D: RCLike + ScalarOperand + Lapack,
{
    fn ehess_to_rhess(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
        egrad: &Self::AmbientPoint,
        ehess: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        let projected_hess = self.projection(point, ehess);
        let xtg = point.t().dot(egrad);
        projected_hess - tangent_vector.dot(&xtg)
    }
}

impl<D> RandomPoint for Grassmann<D>
where
    D: RCLike + ScalarOperand + Lapack,
    StandardNormal: Distribution<D>,
{
    random_point_forward!(StandardNormal);

    fn random_point_impl<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
    where
        Dist: Distribution<Self::Field>,
        R: Rng + ?Sized,
    {
        let point = Array2::random_using((self.n, self.p), &dist, rng);
        let (u, _, vt) = thin_svd_owned(point, SVDBackend::GESDD, None);
        u.dot(&vt)
    }
}

impl<D> Exp for Grassmann<D>
where
    D: RCLike + ScalarOperand + Lapack,
{
    fn exp(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point {
        let (u, s, vt) = thin_svd_ref(tangent_vector, SVDBackend::GESDD, None);
        let s = s.map(|x| D::from(*x).unwrap());
        let cos_s = Array::from_diag(&s.mapv(<D as ComplexFloat>::cos));
        let sin_s = Array::from_diag(&s.mapv(<D as ComplexFloat>::sin));
        point.dot(&vt.t().dot(&cos_s).dot(&vt)) + u.dot(&sin_s).dot(&vt)
    }
}
