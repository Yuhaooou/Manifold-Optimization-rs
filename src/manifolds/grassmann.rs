use ndarray::{ScalarOperand, prelude::*};
use ndarray_rand::RandomExt;
use num_complex::ComplexFloat;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::linalg::{LapackElem, LinalgSVD};
use crate::manifolds::{EGradToRGrad, EHessToRHess, Exp, Manifold, RandomPoint};
use crate::random_point_forward;
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
    D: RCLike + ScalarOperand + LapackElem,
{
    type Point = Array2<D>;
    type TangentVector = Array2<D>;
    type AmbientPoint = Array2<D>;
    type Field = D;

    fn base_point(&self) -> Array2<D> {
        todo!()
    }

    fn zero_tangent_vector(&self, _point: &Array2<D>) -> Array2<D> {
        todo!()
    }

    fn inner(
        &self,
        _point: &Array2<D>,
        tangent_vector1: &Array2<D>,
        tangent_vector2: &Array2<D>,
    ) -> D::Real {
        tangent_vector1.inner(tangent_vector2)
    }

    fn projection(&self, point: &Array2<D>, ambient: &Array2<D>) -> Array2<D> {
        ambient - point.dot(&point.t().dot(ambient))
    }

    fn retraction(&self, point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D> {
        let (u, _, vt) = (point + tangent_vector).into_svd();
        u.dot(&vt)
    }
}

impl<D> EGradToRGrad for Grassmann<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    fn egrad_to_rgrad(&self, point: &Array2<D>, egrad: &Array2<D>) -> Array2<D> {
        self.projection(point, egrad)
    }
}

impl<D> EHessToRHess for Grassmann<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    fn ehess_to_rhess(
        &self,
        point: &Array2<D>,
        tangent_vector: &Array2<D>,
        egrad: &Array2<D>,
        ehess: &Array2<D>,
    ) -> Array2<D> {
        let projected_hess = self.projection(point, ehess);
        let xtg = point.t().dot(egrad);
        projected_hess - tangent_vector.dot(&xtg)
    }
}

impl<D> RandomPoint for Grassmann<D>
where
    D: RCLike + ScalarOperand + LapackElem,
    StandardNormal: Distribution<D::Real>,
{
    random_point_forward!(StandardNormal);

    fn random_point_impl<Dist, R>(&self, dist: Dist, rng: &mut R) -> Array2<D>
    where
        Dist: Distribution<D::Real>,
        R: Rng + ?Sized,
    {
        let point = Array2::random_using((self.n, self.p), &dist, rng).mapv(D::from_real);
        let (u, _, vt) = point.into_svd();
        u.dot(&vt)
    }
}

impl<D> Exp for Grassmann<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    fn exp(&self, point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D> {
        let (u, s, vt) = tangent_vector.svd(false);
        let s = s.map(|x| D::from(*x).unwrap());
        let cos_s = Array::from_diag(&s.mapv(<D as ComplexFloat>::cos));
        let sin_s = Array::from_diag(&s.mapv(<D as ComplexFloat>::sin));
        point.dot(&vt.t().dot(&cos_s).dot(&vt)) + u.dot(&sin_s).dot(&vt)
    }
}
