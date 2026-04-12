use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold, RandomPoint};

#[derive(Debug, Clone)]
/// Optimization problem coupling a manifold and an objective function.
pub struct Problem<'a, M, F, G = (), H = ()>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    manifold: &'a M,
    function: F,
    gradient: G,
    hessian: H,
    init_point: M::Point,
}

impl<'a, M, F> Problem<'a, M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    /// Create a new optimization problem.
    pub fn new_with_init_point(manifold: &'a M, function: F, init_point: M::Point) -> Self {
        Problem {
            manifold,
            function,
            gradient: (),
            hessian: (),
            init_point,
        }
    }
}

impl<'a, M, F> Problem<'a, M, F>
where
    M: Manifold + RandomPoint,
    F: Fn(&M::Point) -> M::Field,
{
    /// Create a new optimization problem.
    pub fn new(manifold: &'a M, function: F) -> Self {
        let init_point = manifold.random_point();
        Problem {
            manifold,
            function,
            gradient: (),
            hessian: (),
            init_point,
        }
    }
}

impl<'a, M, F, G, H> Problem<'a, M, F, G, H>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    pub fn with_rgrad<NG>(self, g: NG) -> Problem<'a, M, F, NG, H>
    where
        NG: Fn(&M::Point) -> M::TangentVector,
    {
        Problem {
            manifold: self.manifold,
            function: self.function,
            gradient: g,
            hessian: self.hessian,
            init_point: self.init_point,
        }
    }

    pub fn with_rhess<NH>(self, h: NH) -> Problem<'a, M, F, G, NH>
    where
        NH: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
    {
        Problem {
            manifold: self.manifold,
            function: self.function,
            gradient: self.gradient,
            hessian: h,
            init_point: self.init_point,
        }
    }

    /// Get the configured initial point.
    pub fn get_initial_point(&self) -> &M::Point {
        &self.init_point
    }

    /// Set a new initial point, returning the old one.
    pub fn set_new_initial_point(&mut self, new_init: M::Point) -> M::Point {
        std::mem::replace(&mut self.init_point, new_init)
    }

    pub fn value(&self, x: &M::Point) -> M::Field {
        (self.function)(x)
    }

    /// Norm induced by manifold metric.
    pub fn norm(&self, x: &M::Point, v: &M::TangentVector) -> M::Field {
        self.manifold.norm(x, v)
    }

    /// Manifold inner product of tangent vectors.
    pub fn inner(&self, x: &M::Point, v1: &M::TangentVector, v2: &M::TangentVector) -> M::Field {
        self.manifold.inner(x, v1, v2)
    }

    /// Retract a tangent vector back to the manifold.
    pub fn retraction(&self, x: &M::Point, u: &M::TangentVector) -> M::Point {
        self.manifold.retraction(x, u)
    }

    /// Project an ambient vector to tangent space.
    pub fn projection(&self, x: &M::Point, u: &M::AmbientPoint) -> M::TangentVector {
        self.manifold.projection(x, u)
    }
}

impl<'a, M, F, G, H> Problem<'a, M, F, G, H>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
    G: Fn(&M::Point) -> M::TangentVector,
{
    pub fn gradient(&self, point: &M::Point) -> M::TangentVector {
        (self.gradient)(point)
    }
}

impl<'a, M, F, G, H> Problem<'a, M, F, G, H>
where
    M: Manifold + EGradToRGrad,
    F: Fn(&M::Point) -> M::Field,
{
    pub fn with_egrad<EG>(
        self,
        g: EG,
    ) -> Problem<'a, M, F, impl Fn(&M::Point) -> M::TangentVector, H>
    where
        EG: Fn(&M::Point) -> M::AmbientPoint,
    {
        let gradient = move |x: &M::Point| self.manifold.egrad_to_rgrad(x, &g(x));

        Problem {
            manifold: self.manifold,
            init_point: self.init_point,
            function: self.function,
            gradient,
            hessian: self.hessian,
        }
    }
}

impl<'a, M, F, G, H> Problem<'a, M, F, G, H>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
    H: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
{
    pub fn hessian(&self, point: &M::Point, tangent_vector: &M::TangentVector) -> M::TangentVector {
        (self.hessian)(point, tangent_vector)
    }
}

impl<'a, M, F, G, H> Problem<'a, M, F, G, H>
where
    M: Manifold + EGradToRGrad + EHessToRHess,
    F: Fn(&M::Point) -> M::Field,
{
    pub fn with_egrad_ehess<EG, EH>(
        self,
        g: EG,
        h: EH,
    ) -> Problem<
        'a,
        M,
        F,
        impl Fn(&M::Point) -> M::TangentVector,
        impl Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
    >
    where
        EG: Fn(&M::Point) -> M::AmbientPoint + Clone,
        EH: Fn(&M::Point, &M::TangentVector) -> M::AmbientPoint,
    {
        let g1 = g.clone();
        let gradient = move |x: &M::Point| self.manifold.egrad_to_rgrad(x, &g(x));

        let hessian = move |x: &M::Point, u: &M::TangentVector| {
            let egrad = g1(x);
            let ehess = h(x, u);
            self.manifold.ehess_to_rhess(x, u, &egrad, &ehess)
        };

        Problem {
            manifold: self.manifold,
            init_point: self.init_point,
            function: self.function,
            gradient,
            hessian,
        }
    }
}
