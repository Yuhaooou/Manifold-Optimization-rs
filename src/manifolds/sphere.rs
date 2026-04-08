use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::{Lapack, Norm, Scalar};
use ndarray_rand::RandomExt;
use rand::{
    Rng,
    distr::{
        Distribution,
        uniform::{SampleUniform, Uniform},
    },
};

use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold, RandomPoint};
use crate::utils::{
    inner_product::InnerProduct,
    traits::{RCLike, Real},
};

#[derive(Debug, Clone)]
/// Sphere manifold $S^{n-1}$ embedded in Euclidean space.
pub struct Sphere<D>
where
    D: RCLike,
{
    name: String,
    n: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D> Sphere<D>
where
    D: RCLike,
{
    /// Create a sphere manifold with ambient dimension `n`.
    pub fn new(n: usize) -> Self {
        Sphere {
            name: format!("Sphere manifold S({})", n),
            n,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D> Manifold for Sphere<D>
where
    D: Real + ScalarOperand + Scalar,
{
    type Field = D;
    type Point = Array1<D>;
    type TangentVector = Array1<D>;
    type AmbientPoint = Array1<D>;

    fn base_point(&self) -> Self::Point {
        let mut res = Array1::zeros(self.n);
        res[0] = D::one();
        res
    }

    fn zero_tangent_vector(&self, _point: &Self::Point) -> Self::TangentVector {
        Array1::zeros(self.n)
    }

    #[inline]
    fn inner(
        &self,
        _point: &Self::Point,
        tangent_vector1: &Self::TangentVector,
        tangent_vector2: &Self::TangentVector,
    ) -> D {
        tangent_vector1.inner(tangent_vector2)
    }

    fn projection(&self, point: &Array1<D>, ambient_point: &Array1<D>) -> Array1<D> {
        ambient_point - point * point.dot(ambient_point)
    }

    fn retraction(&self, point: &Array1<D>, tangent_vector: &Array1<D>) -> Array1<D> {
        let res = point + tangent_vector;
        &res / <D as Scalar>::sqrt(res.dot(&res))
    }
}

impl<D> EGradToRGrad for Sphere<D>
where
    D: Real + ScalarOperand + Scalar,
{
    fn egrad_to_rgrad(&self, point: &Array1<D>, egrad: &Array1<D>) -> Array1<D> {
        self.projection(point, egrad)
    }
}

impl<D> EHessToRHess for Sphere<D>
where
    D: Real + ScalarOperand + Scalar,
{
    fn ehess_to_rhess(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
        egrad: &Self::AmbientPoint,
        ehess: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        self.projection(point, ehess) - tangent_vector * point.dot(egrad)
    }
}

impl<D> RandomPoint for Sphere<D>
where
    D: Real + Scalar + ScalarOperand + SampleUniform + Lapack,
{
    /// Sample a random point and normalize it onto the sphere.
    fn random_point(&self) -> Array1<D> {
        self.random_point_with(Uniform::new(-D::one(), D::one()).unwrap(), &mut rand::rng())
    }

    fn random_point_with_rng<R>(&self, rng: &mut R) -> Self::Point
    where
        R: Rng + ?Sized,
    {
        self.random_point_with(Uniform::new(-D::one(), D::one()).unwrap(), rng)
    }

    fn random_point_with_dist<Dist>(&self, dist: Dist) -> Self::Point
    where
        Dist: Distribution<Self::Field>,
    {
        self.random_point_with(dist, &mut rand::rng())
    }

    fn random_point_with<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
    where
        Dist: Distribution<Self::Field>,
        R: Rng + ?Sized,
    {
        let point = Array1::random_using(self.n, dist, rng);
        &point / D::from_real(point.norm_l2())
    }
}

#[test]
fn test_sphere() {
    let r = 12;
    let eps = 1e-12;
    let sphere = Sphere::<f64>::new(r);

    let point = sphere.random_point();
    assert!((point.norm_l2() - 1.).abs() < eps);

    let ambient_point = Array1::random(r, Uniform::new(0., 1.).unwrap());
    let tangent_vector = sphere.projection(&point, &ambient_point);
    assert!(point.dot(&tangent_vector).abs() < eps);

    let retraction_point = sphere.retraction(&point, &(1.8 * tangent_vector));
    assert!((retraction_point.norm_l2() - 1.).abs() < eps);
}
