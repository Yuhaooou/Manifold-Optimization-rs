use num_complex::ComplexFloat;
use rand::{Rng, distr::Distribution};

use crate::utils::traits::{RCLike, Vector};

/// Generic manifold interface used by optimization algorithms.
/// The reason for methods like [`Self::inner`] needs to be method instead of associated functions
/// is that it may choose which inner product to use by the field in the manifold struct.
pub trait Manifold: Clone {
    /// Scalar type used for geometry computations on this manifold.
    type Field: RCLike;

    /// Point type on the manifold.
    type Point: Clone;

    /// Tangent vector type at each point.
    type TangentVector: Vector<Field = Self::Field>;

    /// Ambient-space representation type used for projections and Euclidean derivatives.
    type AmbientPoint: Clone;

    /// Return a specific point on the manifold.
    fn base_point(&self) -> Self::Point;

    /// Return the zero tangent vector at `point`.
    fn zero_tangent_vector(&self, point: &Self::Point) -> Self::TangentVector;

    /// Convert ambient vector to a point on the manifold. This method may have no theoretical meaning.
    fn to_manifold(&self, ambient: &Self::AmbientPoint) -> Self::Point {
        let _ = ambient;
        unimplemented!("to_manifold is not implemented for this manifold");
    }

    /// Riemannian metric (inner product) at `point`.
    fn inner(
        &self,
        point: &Self::Point,
        tangent_vector1: &Self::TangentVector,
        tangent_vector2: &Self::TangentVector,
    ) -> Self::Field;

    /// Norm induced by [`Self::inner`].
    fn norm(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Field {
        self.inner(point, tangent_vector, tangent_vector).sqrt()
    }

    /// Project ambient vector `ambient` to the tangent space at `point`.
    fn projection(&self, point: &Self::Point, ambient: &Self::AmbientPoint) -> Self::TangentVector;

    /// Retraction map from tangent space back to manifold.
    fn retraction(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point;
}

pub trait EGradToRGrad: Manifold {
    /// Convert Euclidean gradient to Riemannian gradient.
    fn egrad_to_rgrad(
        &self,
        point: &Self::Point,
        egrad: &Self::AmbientPoint,
    ) -> Self::TangentVector;
}

pub trait EHessToRHess: Manifold {
    /// Convert Euclidean Hessian-vector product to Riemannian Hessian-vector product.
    fn ehess_to_rhess(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
        egrad: &Self::AmbientPoint,
        ehess: &Self::AmbientPoint,
    ) -> Self::TangentVector;
}

pub trait Exp: Manifold {
    /// Exponential map (exact geodesic step), when available.
    fn exp(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point;
}

pub trait Log: Manifold {
    /// Logarithm map (inverse of exponential map), when available.
    fn log(&self, point: &Self::Point, other: &Self::Point) -> Self::TangentVector;
}

pub trait Transport: Manifold {
    /// Parallel transport of tangent vector `v` from `point` to `other`.
    fn transport(
        &self,
        point: &Self::Point,
        next_point: &Self::Point,
        tangent_vector: &Self::TangentVector,
    ) -> Self::TangentVector;
}

// TODO: Simplify this trait.
/// Trait for manifolds that can sample random points.
pub trait RandomPoint: Manifold {
    /// Generate a random point on the manifold.
    fn random_point(&self) -> Self::Point;

    fn random_point_with_rng<R>(&self, rng: &mut R) -> Self::Point
    where
        R: Rng + ?Sized;

    fn random_point_with_dist<Dist>(&self, dist: Dist) -> Self::Point
    where
        Dist: Distribution<Self::Field>;

    fn random_point_impl<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
    where
        Dist: Distribution<Self::Field>,
        R: Rng + ?Sized;
}

#[macro_export]
macro_rules! random_point_forward {
    ($dist:expr) => {
        fn random_point(&self) -> Self::Point {
            self.random_point_impl($dist, &mut rand::rng())
        }

        fn random_point_with_rng<R>(&self, rng: &mut R) -> Self::Point
        where
            R: Rng + ?Sized,
        {
            self.random_point_impl($dist, rng)
        }

        fn random_point_with_dist<Dist>(&self, dist: Dist) -> Self::Point
        where
            Dist: Distribution<D>,
        {
            self.random_point_impl(dist, &mut rand::rng())
        }
    };
}

pub trait RandomTangentVector: Manifold {
    /// Generate a random tangent vector at `point`.
    fn random_tangent_vector(&self, point: &Self::Point) -> Self::TangentVector;

    fn random_tangent_vector_with_rng<R>(
        &self,
        point: &Self::Point,
        rng: &mut R,
    ) -> Self::TangentVector
    where
        R: Rng + ?Sized;

    fn random_tangent_vector_with_dist<Dist>(
        &self,
        point: &Self::Point,
        dist: Dist,
    ) -> Self::TangentVector
    where
        Dist: Distribution<Self::Field>;

    fn random_tangent_vector_impl<Dist, R>(
        &self,
        point: &Self::Point,
        dist: Dist,
        rng: &mut R,
    ) -> Self::TangentVector
    where
        Dist: Distribution<Self::Field>,
        R: Rng + ?Sized;
}
