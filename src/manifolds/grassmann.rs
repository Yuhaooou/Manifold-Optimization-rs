use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::{Lapack, Norm, SVD, Scalar};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::manifolds::Manifold;
use crate::manifolds::manifold::{EGradToRGrad, EHessToRHess, Exp, RandomPoint};
use crate::utils::traits::Real;
use crate::utils::{inner_product::InnerProduct, tools::get_scalar_from_float};

#[derive(Debug, Clone)]
/// Grassmann manifold `Gr(n, p)` of `p`-dimensional subspaces in `R^n`.
pub struct Grassmann<D>
where
    D: Real + Scalar + ScalarOperand,
{
    name: String,
    n: usize,
    p: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D> Grassmann<D>
where
    D: Real + Scalar + ScalarOperand,
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

    /// Return a typical distance scale used in trust-region methods.
    pub fn typical_dist(&self) -> D
    where
        D: Real,
    {
        <D as Scalar>::sqrt(get_scalar_from_float::<D>((self.p) as f64))
    }

    /// Geodesic distance approximation via principal angles.
    pub fn dist(&self, point_a: &Array2<D>, point_b: &Array2<D>) -> D
    where
        D: Real + Lapack<Real = D>,
    {
        let s = point_a.t().dot(point_b).svd(false, false).unwrap().1;
        let principal_angles = s.mapv(|x| Scalar::acos(D::min(x, D::one())));
        principal_angles.norm_l2()
    }

    // /// Sample a random unit-norm tangent vector at `point`.
    // pub fn random_tangent_vector(&self, point: &Array2<D>) -> Array2<D>
    // where
    //     D: Real + Lapack<Real = D> + SampleUniform,
    // {
    //     let distribution = Uniform::new(
    //         get_scalar_from_float::<D>(0.),
    //         get_scalar_from_float::<D>(1.),
    //     )
    //     .unwrap();
    //     let tangent_vector =
    //         self.projection(point, &Array2::random((self.n, self.p), distribution));
    //     let tangent_norm = tangent_vector.norm_l2();
    //     tangent_vector / tangent_norm
    // }
}

impl<D> Manifold for Grassmann<D>
where
    D: Scalar + ScalarOperand + Real + Lapack<Real = D>,
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
        let (u, _, vt) = (point + tangent_vector).svd(true, true).unwrap();
        u.unwrap().dot(&vt.unwrap())
    }
}

impl<D> EGradToRGrad for Grassmann<D>
where
    D: Scalar + ScalarOperand + Real + Lapack<Real = D>,
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
    D: Scalar + ScalarOperand + Real + Lapack<Real = D>,
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
    D: Scalar + ScalarOperand + Real + Lapack<Real = D>,
    StandardNormal: Distribution<D>,
{
    /// Sample a random point and project it to `Gr(n, p)` via SVD.
    fn random_point(&self) -> Array2<D> {
        self.random_point_with(StandardNormal, &mut rand::rng())
    }

    fn random_point_with_rng<R>(&self, rng: &mut R) -> Self::Point
    where
        R: Rng + ?Sized,
    {
        self.random_point_with(StandardNormal, rng)
    }

    fn random_point_with_dist<Dist>(&self, dist: Dist) -> Self::Point
    where
        Dist: Distribution<D>,
    {
        self.random_point_with(dist, &mut rand::rng())
    }

    fn random_point_with<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
    where
        Dist: Distribution<Self::Field>,
        R: Rng + ?Sized,
    {
        loop {
            let point = Array2::random_using((self.n, self.p), &dist, rng);
            match point.svd(true, true) {
                Ok((Some(u), _, Some(vt))) => return u.dot(&vt),
                _ => println!(
                    "Warning: get random point failed due to SVD decomposition failure. Retrying..."
                ),
            }
        }
    }
}

impl<D> Exp for Grassmann<D>
where
    D: Scalar + ScalarOperand + Real + Lapack<Real = D>,
{
    fn exp(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point {
        let (u, s, vt) = tangent_vector.svd(true, true).unwrap();
        let u = u.unwrap();
        let vt = vt.unwrap();
        let cos_s = Array::from_diag(&s.mapv(Scalar::cos));
        let sin_s = Array::from_diag(&s.mapv(Scalar::sin));
        point.dot(&vt.t().dot(&cos_s).dot(&vt)) + u.dot(&sin_s).dot(&vt)
    }
}
