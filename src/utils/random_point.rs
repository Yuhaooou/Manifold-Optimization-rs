use crate::manifolds::Manifold;

/// Trait for manifolds that can sample random points.
pub trait RandomOn: Manifold {
    /// Generate a random point on the manifold.
    fn random_point(&self) -> Self::Point;
}
