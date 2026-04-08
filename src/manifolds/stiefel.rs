use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::{Lapack, SVD};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold, RandomPoint};
use crate::utils::{
    inner_product::InnerProduct,
    tools::{mat_sym, qr},
    traits::{RCLike, Real},
};

#[derive(Debug, Clone)]
pub enum RetractionType {
    QR,
    Polar,
}

#[derive(Debug, Clone)]
/// Stiefel manifold `St(n, p)` of orthonormal `n x p` matrices.
pub struct Stiefel<D>
where
    D: RCLike,
{
    name: String,
    n: usize,
    p: usize,
    retraction_type: RetractionType,
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
            retraction_type: RetractionType::QR,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_retraction_type(&mut self, t: RetractionType) -> &mut Self {
        self.retraction_type = t;
        self
    }

    fn retraction_qr(point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D>
    where
        D: Lapack + Real,
    {
        qr(&(point + tangent_vector)).unwrap()
    }

    fn retraction_polar(point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D>
    where
        D: Lapack + Real,
    {
        let (u, _, vt) = (point + tangent_vector).svd(true, true).unwrap();
        u.unwrap().dot(&vt.unwrap())
    }
}

impl<D> Manifold for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack,
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
        tmp1 + point.dot(&tmp2) / D::fromi8(2)
    }

    fn retraction(&self, point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D> {
        match self.retraction_type {
            RetractionType::QR => Self::retraction_qr(point, tangent_vector),
            RetractionType::Polar => Self::retraction_polar(point, tangent_vector),
        }
    }
}

impl<D> EGradToRGrad for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack,
{
    fn egrad_to_rgrad(&self, point: &Array2<D>, egrad: &Array2<D>) -> Array2<D> {
        egrad - point.dot(&mat_sym(&point.t().dot(egrad)))
    }
}

impl<D> EHessToRHess for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack,
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

// Q: Is standard normal randomly enough after QR?
impl<D> RandomPoint for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack,
    StandardNormal: Distribution<D>,
{
    /// Sample a random matrix and retract to the manifold.
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
            match qr(&point) {
                Ok(q) => return q,
                Err(_) => println!(
                    "Warning: get random point failed due to QR decomposition failure. Retrying..."
                ),
            }
        }
    }
}

#[allow(unused_imports)]
mod tests {
    use super::*;
    use ndarray_linalg::Norm;
    use rand::distr::Uniform;

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
