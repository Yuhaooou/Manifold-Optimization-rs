use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::Lapack;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold, RandomPoint};
use crate::utils::{
    tools::{mat_sym, qr, tsvd},
    traits::{InnerProduct, RCLike, Real},
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Retraction type for Stiefel manifold.
pub enum StRetrType {
    QR,
    Polar,
}

#[derive(Debug, Clone)]
/// Stiefel manifold `St(n, p)` of orthonormal `n x p` matrices.
pub struct Stiefel<D>
where
    D: RCLike,
{
    pub name: String,
    n: usize,
    p: usize,
    retraction_type: StRetrType,
    _marker: std::marker::PhantomData<D>,
}

impl<D> Stiefel<D>
where
    D: RCLike + Lapack<Real = D>,
{
    /// Create `St(n, p)` with default QR retraction.
    pub fn new(n: usize, p: usize) -> Self {
        assert!(n >= p && p >= 1, "Need n >= p >= 1");
        Stiefel {
            name: format!("Stiefel manifold St({},{})", n, p),
            n,
            p,
            retraction_type: StRetrType::QR,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set retraction type by enum.
    pub fn set_retraction(mut self, t: StRetrType) -> Self {
        self.retraction_type = t;
        self
    }

    /// Set retraction type by u8: 0 for QR, 1 for Polar.
    pub fn set_retraction_u8(mut self, t: u8) -> Self {
        self.retraction_type = match t {
            0 => StRetrType::QR,
            1 => StRetrType::Polar,
            _ => panic!("Invalid retraction type index: {t}"),
        };
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
        D: Lapack<Real = D> + Real,
    {
        let (u, _, vt) = tsvd(&(point + tangent_vector)).unwrap();
        u.dot(&vt)
    }

    pub fn set_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

impl<D> Manifold for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack<Real = D>,
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
        tmp1 + point.dot(&tmp2) / D::from_i8(2).unwrap()
    }

    fn retraction(&self, point: &Array2<D>, tangent_vector: &Array2<D>) -> Array2<D> {
        match self.retraction_type {
            StRetrType::QR => Self::retraction_qr(point, tangent_vector),
            StRetrType::Polar => Self::retraction_polar(point, tangent_vector),
        }
    }
}

impl<D> EGradToRGrad for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack<Real = D>,
{
    fn egrad_to_rgrad(&self, point: &Array2<D>, egrad: &Array2<D>) -> Array2<D> {
        egrad - point.dot(&mat_sym(&point.t().dot(egrad)))
    }
}

impl<D> EHessToRHess for Stiefel<D>
where
    D: Real + ScalarOperand + Lapack<Real = D>,
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
    D: Real + ScalarOperand + Lapack<Real = D>,
    StandardNormal: Distribution<D>,
{
    /// Sample a random matrix and retract to the manifold.
    fn random_point(&self) -> Array2<D> {
        self.random_point_impl(StandardNormal, &mut rand::rng())
    }

    fn random_point_with_rng<R>(&self, rng: &mut R) -> Self::Point
    where
        R: Rng + ?Sized,
    {
        self.random_point_impl(StandardNormal, rng)
    }

    fn random_point_with_dist<Dist>(&self, dist: Dist) -> Self::Point
    where
        Dist: Distribution<D>,
    {
        self.random_point_impl(dist, &mut rand::rng())
    }

    fn random_point_impl<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
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
