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

    /// Return a canonical base point on the manifold.
    fn base_point(&self) -> Self::Point {
        unimplemented!("Base point not implemented for this manifold");
    }

    /// Return the zero tangent vector at `point`.
    fn zero_tangent_vector(&self, point: &Self::Point) -> Self::TangentVector {
        let _ = point;
        unimplemented!("Zero tangent vector not implemented for this manifold");
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
        (self.inner(point, tangent_vector, tangent_vector)).sqrt()
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

pub trait Exponential: Manifold {
    /// Exponential map (exact geodesic step), when available.
    fn exponential_map(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
    ) -> Self::Point;
}
