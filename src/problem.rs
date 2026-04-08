use crate::manifolds::Manifold;
use crate::manifolds::manifold::{EGradToRGrad, EHessToRHess};
use crate::utils::random_point::RandomOn;
use crate::utils::traits::RCLike;

/// Objective function defined on a manifold point.
pub trait Function {
    /// Point type for function input.
    type Point;

    /// Scalar output type of the objective.
    type Field: RCLike;

    /// Evaluate objective value at `x`.
    fn value(&self, x: &Self::Point) -> Self::Field;
}

/// Euclidean gradient provider for an objective.
pub trait EGradient {
    /// Point type for gradient evaluation.
    type Point;

    /// Euclidean gradient type.
    type Gradient;

    /// Compute Euclidean gradient at `x`.
    fn euclidean_gradient(&self, x: &Self::Point) -> Self::Gradient;
}

/// Marker trait for functions with Euclidean gradient.
pub trait FuncWithEGrad<Field, Point, Gradient>:
    Function<Point = Point, Field = Field> + EGradient<Point = Point, Gradient = Gradient>
{
}

impl<Field, Point, Gradient, T> FuncWithEGrad<Field, Point, Gradient> for T where
    T: Function<Point = Point, Field = Field> + EGradient<Point = Point, Gradient = Gradient>
{
}

/// Euclidean Hessian provider for an objective.
pub trait EHessian {
    /// Point type for Hessian evaluation.
    type Point;

    /// Direction type for Hessian-vector products.
    type Direction;

    /// Output type of Hessian action.
    type Hessian;

    /// Compute Euclidean Hessian action at `x` along direction `u`.
    fn euclidean_hessian(&self, x: &Self::Point, u: &Self::Direction) -> Self::Hessian;
}

/// Marker trait for functions with Euclidean Hessian.
pub trait FuncWithEHess<Field, Point, Direction, Hessian>:
    Function<Point = Point, Field = Field>
    + EHessian<Point = Point, Direction = Direction, Hessian = Hessian>
{
}

impl<Field, Point, Direction, Hessian, T> FuncWithEHess<Field, Point, Direction, Hessian> for T where
    T: Function<Point = Point, Field = Field>
        + EHessian<Point = Point, Direction = Direction, Hessian = Hessian>
{
}

/// Marker trait for functions with both Euclidean gradient and Hessian.
pub trait FuncWithEGradEHess<Field, Point, Gradient, Direction, Hessian>:
    Function<Point = Point, Field = Field>
    + EGradient<Point = Point, Gradient = Gradient>
    + EHessian<Point = Point, Direction = Direction, Hessian = Hessian>
{
}

impl<Field, Point, Gradient, Direction, Hessian, T>
    FuncWithEGradEHess<Field, Point, Gradient, Direction, Hessian> for T
where
    T: Function<Point = Point, Field = Field>
        + EGradient<Point = Point, Gradient = Gradient>
        + EHessian<Point = Point, Direction = Direction, Hessian = Hessian>,
{
}

#[derive(Debug, Clone)]
/// Optimization problem coupling a manifold and an objective function.
pub struct Problem<'a, M, F>
where
    M: Manifold,
    F: Function<Point = M::Point>,
{
    pub(crate) manifold: &'a M,
    pub(crate) function: &'a F,
    pub(crate) initial_point: Option<M::Point>,
}

impl<'a, M, F> Problem<'a, M, F>
where
    M: Manifold,
    F: Function<Point = M::Point>,
{
    /// Create a new optimization problem.
    pub fn new(manifold: &'a M, function: &'a F) -> Self {
        Problem {
            manifold,
            function,
            initial_point: None,
        }
    }
}

impl<M, F> Problem<'_, M, F>
where
    M: Manifold,
    F: Function<Point = M::Point>,
{
    /// Set the initial point for optimization.
    ///
    /// This is optional when the manifold also implements `RandomOn`.
    pub fn set_initial_point(mut self, init_point: M::Point) -> Self {
        println!("Set initial point.");
        self.initial_point = Some(init_point);
        self
    }

    /// Get the configured initial point.
    ///
    /// # Panics
    /// Panics if initial point has not been set.
    pub fn get_initial_point(&self) -> &M::Point {
        self.initial_point
            .as_ref()
            .expect("Initial point is not set")
    }

    #[inline]
    /// Evaluate objective value at point `x`.
    pub fn value(&self, x: &M::Point) -> F::Field {
        self.function.value(x)
    }

    #[inline]
    /// Norm induced by manifold metric.
    pub fn norm(&self, x: &M::Point, v: &M::TangentVector) -> M::Field {
        self.manifold.norm(x, v)
    }

    #[inline]
    /// Manifold inner product of tangent vectors.
    pub fn inner(&self, x: &M::Point, v1: &M::TangentVector, v2: &M::TangentVector) -> M::Field {
        self.manifold.inner(x, v1, v2)
    }

    #[inline]
    /// Retract a tangent vector back to the manifold.
    pub fn retraction(&self, x: &M::Point, u: &M::TangentVector) -> M::Point {
        self.manifold.retraction(x, u)
    }

    #[inline]
    /// Project an ambient vector to tangent space.
    pub fn projection(&self, x: &M::Point, u: &M::AmbientPoint) -> M::TangentVector {
        self.manifold.projection(x, u)
    }
}

impl<M, F> Problem<'_, M, F>
where
    M: Manifold + EGradToRGrad,
    F: Function<Point = M::Point, Field = M::Field>
        + EGradient<Point = M::Point, Gradient = M::AmbientPoint>,
{
    #[inline]
    /// Euclidean gradient of the objective.
    pub fn euclidean_gradient(&self, x: &M::Point) -> M::AmbientPoint {
        self.function.euclidean_gradient(x)
    }

    #[inline]
    /// Convert Euclidean gradient to Riemannian gradient.
    pub fn egrad_to_rgrad(&self, x: &M::Point, egrad: &M::AmbientPoint) -> M::TangentVector {
        self.manifold.egrad_to_rgrad(x, egrad)
    }

    /// Riemannian gradient of the objective.
    pub fn gradient(&self, x: &M::Point) -> M::TangentVector {
        let egrad = self.euclidean_gradient(x);
        self.egrad_to_rgrad(x, &egrad)
    }
}

impl<M, F> Problem<'_, M, F>
where
    M: Manifold + EGradToRGrad + EHessToRHess,
    F: Function<Point = M::Point, Field = M::Field>
        + EGradient<Point = M::Point, Gradient = M::AmbientPoint>
        + EHessian<Point = M::Point, Direction = M::TangentVector, Hessian = M::AmbientPoint>,
{
    #[inline]
    /// Euclidean Hessian action of the objective.
    pub fn euclidean_hessian(&self, x: &M::Point, u: &M::TangentVector) -> M::AmbientPoint {
        self.function.euclidean_hessian(x, u)
    }

    #[inline]
    /// Convert Euclidean Hessian action to Riemannian Hessian action.
    pub fn ehess_to_rhess(
        &self,
        x: &M::Point,
        u: &M::TangentVector,
        egrad: &M::AmbientPoint,
        ehess: &M::AmbientPoint,
    ) -> M::TangentVector {
        self.manifold.ehess_to_rhess(x, u, egrad, ehess)
    }

    /// Riemannian Hessian action.
    pub fn hessian(&self, x: &M::Point, u: &M::TangentVector) -> M::TangentVector {
        let egrad = self.euclidean_gradient(x);
        let ehess = self.euclidean_hessian(x, u);
        self.ehess_to_rhess(x, u, &egrad, &ehess)
    }
}

impl<M, F> Problem<'_, M, F>
where
    M: Manifold + RandomOn,
    F: Function<Point = M::Point, Field = M::Field>,
{
    // pub fn set_random_initial_point(mut self) -> Self {
    //     self.initial_point = Some(self.manifold.random_point());
    //     self
    // }

    /// Return initial point, or lazily initialize it with a random point.
    ///
    /// Useful for algorithms that support random initialization by default.
    pub fn get_or_init_initial_point(&mut self) -> &M::Point {
        if self.initial_point.is_none() {
            println!("Set initial point.");
            self.initial_point = Some(self.manifold.random_point());
        }
        self.get_initial_point()
    }
}
