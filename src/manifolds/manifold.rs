use crate::utils::traits::{Field, RCLike, Vector};

/// Generic manifold interface used by optimization algorithms.
pub trait Manifold: Clone {
    /// Scalar type used for geometry computations on this manifold.
    type Field: RCLike;

    /// Point type on the manifold.
    type Point: Clone;

    /// Tangent vector type at each point.
    type TangentVector: Vector<Field = Self::Field>;

    /// Ambient-space representation type used for projections and Euclidean derivatives.
    type AmbientPoint: Clone;

    // fn ispoint_on_manifold(&self, point: &Self::Point) -> bool {
    //     println!("ispoint_on_manifold not implemented for this manifold");
    //     false
    // }

    // fn istangent_vector(
    //     &self,
    //     point: &Self::Point,
    //     tangent_vector: &Self::TangentVector,
    // ) -> bool {
    //     println!("istangent_vector not implemented for this manifold");
    //     false
    // }

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
        tangent_vector_a: &Self::TangentVector,
        tangent_vector_b: &Self::TangentVector,
    ) -> <Self::Field as Field>::Real {
        let _ = (point, tangent_vector_a, tangent_vector_b);
        unimplemented!("Metric not implemented for this manifold");
    }

    /// Norm induced by [`Self::inner`].
    fn norm(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
    ) -> <Self::Field as Field>::Real {
        (self.inner(point, tangent_vector, tangent_vector)).sqrt()
    }

    /// Project ambient vector `ambient` to the tangent space at `point`.
    fn projection(&self, point: &Self::Point, ambient: &Self::AmbientPoint) -> Self::TangentVector {
        let _ = (point, ambient);
        unimplemented!("Projection not implemented for this manifold");
    }

    /// Retraction map from tangent space back to manifold.
    fn retraction(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point {
        let _ = (point, tangent_vector);
        unimplemented!("Retraction not implemented for this manifold");
    }

    /// Exponential map (exact geodesic step), when available.
    fn exponential_map(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
    ) -> Self::Point {
        let _ = (point, tangent_vector);
        unimplemented!("Exponential map not implemented for this manifold");
    }

    /// Convert Euclidean gradient to Riemannian gradient.
    fn egrad_to_rgrad(
        &self,
        point: &Self::Point,
        egrad: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        let _ = (point, egrad);
        unimplemented!("Egrad to Rgrad conversion not implemented for this manifold");
    }

    /// Convert Euclidean Hessian-vector product to Riemannian Hessian-vector product.
    fn ehess_to_rhess(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
        egrad: &Self::AmbientPoint,
        ehess: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        let _ = (point, tangent_vector, egrad, ehess);
        unimplemented!("Ehess to Rhess conversion not implemented for this manifold");
    }
}
